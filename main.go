package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	_ "modernc.org/sqlite" // Import the pure Go sqlite3 driver
)

// --- Constants ---
const (
	wsPath              = "/v1/ws"
	proxyListenAddr     = ":5345"
	wsReadTimeout       = 60 * time.Second
	proxyRequestTimeout = 600 * time.Second
)

// --- Timezone ---
var cstZone, _ = time.LoadLocation("Asia/Shanghai") // 中国标准时间

// --- API Usage Statistics ---

const dbFilename = "usage_stats.db"

var db *sql.DB

// UsageData holds statistics for a given key.
type UsageData struct {
	Count            int64   `json:"count"`
	PromptTokens     int64   `json:"prompt_tokens"`
	CandidatesTokens int64   `json:"candidates_tokens"`
	TotalTokens      int64   `json:"total_tokens"`
	Cost             float64 `json:"cost"`
}

// initDB 初始化数据库连接并创建表
func initDB() {
	var err error
	db, err = sql.Open("sqlite", dbFilename)
	if err != nil {
		log.Fatalf("打开数据库失败: %v", err)
	}

	createTableSQL := `CREATE TABLE IF NOT EXISTS usage_stats (
		"model_name" TEXT NOT NULL,
		"date" TEXT NOT NULL,
		"count" BIGINT NOT NULL DEFAULT 0,
		"prompt_tokens" BIGINT NOT NULL DEFAULT 0,
		"candidates_tokens" BIGINT NOT NULL DEFAULT 0,
		"total_tokens" BIGINT NOT NULL DEFAULT 0,
		"cost" REAL NOT NULL DEFAULT 0.0,
		PRIMARY KEY (model_name, date)
	);`

	_, err = db.Exec(createTableSQL)
	if err != nil {
		log.Fatalf("创建数据表失败: %v", err)
	}
	log.Println("数据库初始化成功。")
}

// --- Pricing ---

// PriceTier 定义了特定token阈值下的价格
type PriceTier struct {
	Threshold  int64   // Token 数量阈值 (大于该值时应用此价格)
	InputCost  float64 // 每100万输入token的成本
	OutputCost float64 // 每100万输出token的成本
}

// ModelPricing 包含一个模型的所有价格层级，按阈值从低到高排序
type ModelPricing struct {
	Tiers []PriceTier
}

var pricing = map[string]ModelPricing{
	// --- Models with Tiered Pricing ---
	"gemini-2.5-pro": {Tiers: []PriceTier{
		{Threshold: 200000, InputCost: 2.50, OutputCost: 15.00},
		{Threshold: 0, InputCost: 1.25, OutputCost: 10.00},
	}},

	// --- Models with Flat Pricing ---
	"gemini-pro":            {Tiers: []PriceTier{{Threshold: 0, InputCost: 1.25, OutputCost: 5.00}}}, // Assume same as 1.5 pro base
	"gemini-2.5-flash":      {Tiers: []PriceTier{{Threshold: 0, InputCost: 0.30, OutputCost: 2.50}}},
	"gemini-2.5-flash-lite": {Tiers: []PriceTier{{Threshold: 0, InputCost: 0.10, OutputCost: 0.40}}},
}

// calculateCost 根据模型的定价层级计算请求成本
func calculateCost(modelName string, promptTokens, candidatesTokens int64) float64 {
	modelPricing, ok := pricing[modelName]
	if !ok {
		// 回退机制：对于不在列表中的模型（例如 gemini-pro-vision），尝试通过移除最后一个部分来查找基础模型价格
		parts := strings.Split(modelName, "-")
		if len(parts) > 1 {
			baseModel := strings.Join(parts[:len(parts)-1], "-")
			modelPricing, ok = pricing[baseModel]
		}
		if !ok {
			log.Printf("警告：找不到模型 '%s' 的定价信息", modelName)
			return 0.0
		}
	}

	// 根据 promptTokens 确定价格层级
	// 价格按阈值从高到低排序，因此第一个匹配的即为正确层级
	var selectedTier PriceTier
	for _, tier := range modelPricing.Tiers {
		if promptTokens > tier.Threshold {
			selectedTier = tier
			break
		}
	}
	// 如果循环后仍未选择（例如 promptTokens 正好等于或小于最低阈值），则选择最后一个层级（阈值为0的层级）
	if selectedTier.Threshold == 0 && len(modelPricing.Tiers) > 0 {
		selectedTier = modelPricing.Tiers[len(modelPricing.Tiers)-1]
	}

	inputCost := (float64(promptTokens) / 1000000) * selectedTier.InputCost
	outputCost := (float64(candidatesTokens) / 1000000) * selectedTier.OutputCost
	return inputCost + outputCost
}

// recordUsage 将使用次数和token数量增加到数据库中
func recordUsage(modelName string, usage *UsageMetadata) {
	if modelName == "" {
		return
	}
	date := time.Now().In(cstZone).Format("2006-01-02")

	var promptTokens, candidatesTokens, totalTokens int64
	if usage != nil {
		promptTokens = int64(usage.PromptTokenCount)
		candidatesTokens = int64(usage.CandidatesTokenCount)
		totalTokens = int64(usage.TotalTokenCount)
	}

	cost := calculateCost(modelName, promptTokens, candidatesTokens)

	upsertSQL := `INSERT INTO usage_stats (model_name, date, count, prompt_tokens, candidates_tokens, total_tokens, cost)
	VALUES (?, ?, 1, ?, ?, ?, ?)
	ON CONFLICT(model_name, date) DO UPDATE SET
		count = count + 1,
		prompt_tokens = prompt_tokens + excluded.prompt_tokens,
		candidates_tokens = candidates_tokens + excluded.candidates_tokens,
		total_tokens = total_tokens + excluded.total_tokens,
		cost = cost + excluded.cost;`

	_, err := db.Exec(upsertSQL, modelName, date, promptTokens, candidatesTokens, totalTokens, cost)
	if err != nil {
		log.Printf("记录使用情况到数据库时出错: %v", err)
	}
}

// recordErrorRequest 记录一个非200的HTTP请求到数据库
func recordErrorRequest(statusCode int) {
	if statusCode == http.StatusOK {
		return
	}
	date := time.Now().In(cstZone).Format("2006-01-02")
	modelName := fmt.Sprintf("error-status-%d", statusCode)

	upsertSQL := `INSERT INTO usage_stats (model_name, date, count)
	VALUES (?, ?, 1)
	ON CONFLICT(model_name, date) DO UPDATE SET
		count = count + 1;`

	_, err := db.Exec(upsertSQL, modelName, date)
	if err != nil {
		log.Printf("记录错误请求到数据库时出错: %v", err)
	}
}

// --- Gemini API Structs ---

type GeminiPart struct {
	Text       string      `json:"text,omitempty"`
	InlineData *InlineData `json:"inline_data,omitempty"`
}

type InlineData struct {
	MimeType string `json:"mime_type"`
	Data     string `json:"data"`
}

type GeminiContent struct {
	Role  string       `json:"role"`
	Parts []GeminiPart `json:"parts"`
}

type GenerationConfig struct {
	Temperature     float64  `json:"temperature,omitempty"`
	TopP            float64  `json:"topP,omitempty"`
	MaxOutputTokens int      `json:"maxOutputTokens,omitempty"`
	StopSequences   []string `json:"stopSequences,omitempty"`
}

type GeminiRequest struct {
	Contents         []GeminiContent   `json:"contents"`
	GenerationConfig *GenerationConfig `json:"generationConfig,omitempty"`
}

type GeminiCandidate struct {
	Content      GeminiContent `json:"content"`
	FinishReason string        `json:"finishReason"`
}

type UsageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

type GeminiResponse struct {
	Candidates    []GeminiCandidate `json:"candidates"`
	UsageMetadata UsageMetadata     `json:"usageMetadata"`
}

// GeminiModelInfo describes a single model from the Gemini API
type GeminiModelInfo struct {
	Name        string `json:"name"`
	DisplayName string `json:"displayName"`
}

// GeminiModelListResponse is the response from the Gemini list models API
type GeminiModelListResponse struct {
	Models []GeminiModelInfo `json:"models"`
}

// --- OpenAI API Structs ---

type OpenAIMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // Can be string or []OpenAIPart
}

type OpenAIPart struct {
	Type     string          `json:"type"`
	Text     string          `json:"text,omitempty"`
	ImageURL *OpenAIImageURL `json:"image_url,omitempty"`
}

type OpenAIImageURL struct {
	URL string `json:"url"`
}

type OpenAIRequest struct {
	Model       string          `json:"model"`
	Messages    []OpenAIMessage `json:"messages"`
	Stream      bool            `json:"stream"`
	Temperature float64         `json:"temperature"`
	TopP        float64         `json:"top_p"`
	MaxTokens   int             `json:"max_tokens"`
	Stop        []string        `json:"stop"`
}

type OpenAIChoice struct {
	Index        int           `json:"index"`
	Message      OpenAIMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type OpenAIResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []OpenAIChoice `json:"choices"`
	Usage   UsageMetadata  `json:"usage"`
}

type OpenAIStreamChoice struct {
	Index        int           `json:"index"`
	Delta        OpenAIMessage `json:"delta"`
	FinishReason *string       `json:"finish_reason"`
}

type OpenAIStreamResponse struct {
	ID      string               `json:"id"`
	Object  string               `json:"object"`
	Created int64                `json:"created"`
	Model   string               `json:"model"`
	Choices []OpenAIStreamChoice `json:"choices"`
}

// --- Connection Status Structs ---

// ConnectionInfo holds details about a single client connection for status reporting.
type ConnectionInfo struct {
	RemoteAddr string    `json:"remote_addr"`
	LastActive time.Time `json:"last_active"`
	Healthy    bool      `json:"healthy"`
}

// UserConnectionStatus holds the status for all connections of a single user.
type UserConnectionStatus struct {
	ConnectionCount int              `json:"connection_count"`
	Connections     []ConnectionInfo `json:"connections"`
}

// --- 1. 连接管理与负载均衡 ---

// UserConnection 存储单个WebSocket连接及其元数据
type UserConnection struct {
	Conn       *websocket.Conn
	UserID     string
	LastActive time.Time
	writeMutex sync.Mutex // 保护对此单个连接的并发写入
	Healthy    bool       // 连接是否健康可用
}

// safeWriteJSON 线程安全地向单个WebSocket连接写入JSON
func (uc *UserConnection) safeWriteJSON(v interface{}) error {
	uc.writeMutex.Lock()
	defer uc.writeMutex.Unlock()
	return uc.Conn.WriteJSON(v)
}

// UserConnections 维护单个用户的所有连接和负载均衡状态
type UserConnections struct {
	sync.Mutex
	Connections []*UserConnection
	NextIndex   int // 用于轮询 (round-robin)
}

// ConnectionPool 全局连接池，并发安全
type ConnectionPool struct {
	sync.RWMutex
	Users map[string]*UserConnections
}

var globalPool = &ConnectionPool{
	Users: make(map[string]*UserConnections),
}

// AddConnection 将新连接添加到池中
func (p *ConnectionPool) AddConnection(userID string, conn *websocket.Conn) *UserConnection {
	userConn := &UserConnection{
		Conn:       conn,
		UserID:     userID,
		LastActive: time.Now(),
		Healthy:    true, // 新连接默认为健康状态
	}

	p.Lock()
	defer p.Unlock()

	userConns, exists := p.Users[userID]
	if !exists {
		userConns = &UserConnections{
			Connections: make([]*UserConnection, 0),
			NextIndex:   0,
		}
		p.Users[userID] = userConns
	}

	userConns.Lock()
	userConns.Connections = append(userConns.Connections, userConn)
	userConns.Unlock()

	log.Printf("WebSocket connected: UserID=%s, Total connections for user: %d", userID, len(userConns.Connections))
	return userConn
}

// RemoveConnection 从池中移除连接
func (p *ConnectionPool) RemoveConnection(userID string, conn *websocket.Conn) {
	p.Lock()
	defer p.Unlock()

	userConns, exists := p.Users[userID]
	if !exists {
		return
	}

	userConns.Lock()
	defer userConns.Unlock()

	// 查找并移除连接
	for i, uc := range userConns.Connections {
		if uc.Conn == conn {
			// 高效删除：将最后一个元素移到当前位置，然后截断切片
			userConns.Connections[i] = userConns.Connections[len(userConns.Connections)-1]
			userConns.Connections = userConns.Connections[:len(userConns.Connections)-1]
			log.Printf("WebSocket disconnected: UserID=%s, Remaining connections for user: %d", userID, len(userConns.Connections))
			break
		}
	}

	// 如果该用户没有连接了，可以从主map中删除用户条目（可选）
	if len(userConns.Connections) == 0 {
		delete(p.Users, userID)
	}
}

// GetConnection 使用轮询策略为用户选择一个连接
func (p *ConnectionPool) GetConnection(userID string) (*UserConnection, error) {
	p.RLock()
	userConns, exists := p.Users[userID]
	p.RUnlock()

	if !exists {
		return nil, errors.New("no available client for this user")
	}

	userConns.Lock()
	defer userConns.Unlock()

	numConns := len(userConns.Connections)
	if numConns == 0 {
		// 理论上如果存在于p.Users中，这里不应该为0，但为了健壮性还是检查
		return nil, errors.New("no available client for this user")
	}

	// 轮询负载均衡
	idx := userConns.NextIndex % numConns
	selectedConn := userConns.Connections[idx]
	userConns.NextIndex = (userConns.NextIndex + 1) % numConns // 更新索引

	return selectedConn, nil
}

// GetAllConnections 获取用户的所有活动连接,并按轮询顺序排列
func (p *ConnectionPool) GetAllConnections(userID string) []*UserConnection {
	p.RLock()
	userConns, exists := p.Users[userID]
	p.RUnlock()

	if !exists {
		return nil
	}

	userConns.Lock()
	defer userConns.Unlock()

	numConns := len(userConns.Connections)
	if numConns == 0 {
		return nil
	}

	// 首先过滤出健康的连接
	healthyConns := make([]*UserConnection, 0)
	for _, conn := range userConns.Connections {
		if conn.Healthy {
			healthyConns = append(healthyConns, conn)
		}
	}

	numHealthyConns := len(healthyConns)
	if numHealthyConns == 0 {
		// 如果没有健康的连接，返回所有连接（降级处理）
		healthyConns = userConns.Connections
		numHealthyConns = numConns
	}

	// 轮询负载均衡：确定起始点，并为下一次调用更新索引
	startIndex := userConns.NextIndex % numHealthyConns
	userConns.NextIndex = (userConns.NextIndex + 1) % numHealthyConns

	// 创建一个从 startIndex 开始的旋转列表，以实现负载均衡
	rotatedConns := make([]*UserConnection, numHealthyConns)
	for i := 0; i < numHealthyConns; i++ {
		rotatedConns[i] = healthyConns[(startIndex+i)%numHealthyConns]
	}

	return rotatedConns
}

// markConnectionUnhealthy 标记连接为不健康状态
func markConnectionUnhealthy(conn *UserConnection) {
	conn.Healthy = false
	log.Printf("标记连接为不健康状态: UserID=%s", conn.UserID)
}

// markConnectionHealthy 标记连接为健康状态
func markConnectionHealthy(conn *UserConnection) {
	conn.Healthy = true
	log.Printf("标记连接为健康状态: UserID=%s", conn.UserID)
}

// recoverUnhealthyConnections 定期恢复不健康的连接
func recoverUnhealthyConnections() {
	globalPool.RLock()
	defer globalPool.RUnlock()

	recoveredCount := 0
	for userID, userConns := range globalPool.Users {
		userConns.Lock()
		for _, conn := range userConns.Connections {
			if !conn.Healthy {
				// 检查连接是否仍然活跃（通过最后活动时间）
				if time.Since(conn.LastActive) < 5*time.Minute {
					conn.Healthy = true
					recoveredCount++
					log.Printf("自动恢复不健康连接: UserID=%s", userID)
				}
			}
		}
		userConns.Unlock()
	}

	if recoveredCount > 0 {
		log.Printf("共恢复了 %d 个不健康连接", recoveredCount)
	}
}

// --- 2. WebSocket 消息结构 & 待处理请求 ---

// WSMessage 是前后端之间通信的基本结构
type WSMessage struct {
	ID      string                 `json:"id"`      // 请求/响应的唯一ID
	Type    string                 `json:"type"`    // ping, pong, http_request, http_response, stream_start, stream_chunk, stream_end, error
	Payload map[string]interface{} `json:"payload"` // 具体数据
}

// pendingRequests 存储待处理的HTTP请求，等待WS响应
// key: reqID (string), value: chan *WSMessage
var pendingRequests sync.Map

// --- 3. WebSocket 处理器和心跳 ---

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	// 生产环境中应设置严格的CheckOrigin
	CheckOrigin: func(r *http.Request) bool { return true },
}

func handleWebSocket(w http.ResponseWriter, r *http.Request) {
	// 认证
	authToken := r.URL.Query().Get("auth_token")
	userID, err := validateJWT(authToken)
	if err != nil {
		log.Printf("WebSocket authentication failed: %v", err)
		recordErrorRequest(http.StatusUnauthorized)
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	// 升级连接
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("Failed to upgrade to WebSocket: %v", err)
		return
	}

	// 添加到连接池
	userConn := globalPool.AddConnection(userID, conn)

	// 启动读取循环
	go readPump(userConn)
}

// readPump 处理来自单个WebSocket连接的所有传入消息
func readPump(uc *UserConnection) {
	defer func() {
		globalPool.RemoveConnection(uc.UserID, uc.Conn)
		uc.Conn.Close()
		log.Printf("readPump closed for user %s", uc.UserID)
	}()

	// 设置读取超时 (心跳机制)
	uc.Conn.SetReadDeadline(time.Now().Add(wsReadTimeout))

	for {
		_, message, err := uc.Conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("WebSocket read error for user %s: %v", uc.UserID, err)
			} else {
				log.Printf("WebSocket closed for user %s: %v", uc.UserID, err)
			}
			// 如果读取失败（包括超时），退出循环并清理连接
			break
		}

		// 收到任何消息，重置读取超时
		uc.Conn.SetReadDeadline(time.Now().Add(wsReadTimeout))
		uc.LastActive = time.Now()
		// 如果连接之前被标记为不健康，现在恢复为健康状态
		if !uc.Healthy {
			markConnectionHealthy(uc)
		}

		// 解析消息
		var msg WSMessage
		if err := json.Unmarshal(message, &msg); err != nil {
			log.Printf("Error unmarshalling WebSocket message: %v", err)
			continue
		}

		switch msg.Type {
		case "ping":
			// 心跳响应
			err := uc.safeWriteJSON(map[string]string{"type": "pong", "id": msg.ID})
			if err != nil {
				log.Printf("Error sending pong: %v", err)
				return // 发送失败，认为连接已断
			}
		case "http_response", "stream_start", "stream_chunk", "stream_end", "error":
			// 路由响应到等待的HTTP Handler
			if ch, ok := pendingRequests.Load(msg.ID); ok {
				respChan := ch.(chan *WSMessage)
				// 尝试发送，如果通道已满（不太可能，但为了安全），则记录日志
				select {
				case respChan <- &msg:
				default:
					log.Printf("Warning: Response channel full for request ID %s, dropping message type %s", msg.ID, msg.Type)
				}
			} else {
				log.Printf("Received response for unknown or timed-out request ID: %s", msg.ID)
			}
		default:
			log.Printf("Received unknown message type from client: %s", msg.Type)
		}
	}
}

// --- 4. HTTP 反向代理与 WS 隧道 ---

// forwardRequestToBrowser 封装了将请求通过WebSocket转发给浏览器的核心逻辑
func forwardRequestToBrowser(w http.ResponseWriter, r *http.Request, userID string, wsPayload WSMessage, modelName string) {
	allConns := globalPool.GetAllConnections(userID)
	if len(allConns) == 0 {
		log.Printf("No connections available for user %s", userID)
		recordErrorRequest(http.StatusServiceUnavailable)
		http.Error(w, "Service Unavailable: No active client connected", http.StatusServiceUnavailable)
		return
	}

	var lastError error

	for i, conn := range allConns {
		log.Printf("Attempting request %s on connection %d/%d for user %s", wsPayload.ID, i+1, len(allConns), userID)
		respChan := make(chan *WSMessage, 10)
		pendingRequests.Store(wsPayload.ID, respChan)

		if err := conn.safeWriteJSON(wsPayload); err != nil {
			log.Printf("Failed to send request over WebSocket on attempt %d: %v", i+1, err)
			lastError = err
			pendingRequests.Delete(wsPayload.ID) // 清理
			// 标记连接为不健康
			markConnectionUnhealthy(conn)
			continue // 尝试下一个连接
		}

		// 等待初始响应
		ctx, cancel := context.WithTimeout(r.Context(), proxyRequestTimeout)
		select {
		case msg := <-respChan:
			// 检查是否是成功的初始消息
			isSuccess := false
			if msg.Type == "http_response" {
				if status, ok := msg.Payload["status"].(float64); ok && int(status) == http.StatusOK {
					isSuccess = true
				}
			} else if msg.Type == "stream_start" {
				isSuccess = true
			}

			if isSuccess {
				log.Printf("Request %s succeeded on connection %d", wsPayload.ID, i+1)
				// 成功，标记连接为健康状态
				markConnectionHealthy(conn)
				// 处理剩余的响应
				processWebSocketResponse(w, r, respChan, modelName, msg) // 传入初始消息
				cancel()
				pendingRequests.Delete(wsPayload.ID)
				return
			}
			// 收到的是错误或非200响应
			log.Printf("Attempt %d for request %s failed with message type %s", i+1, wsPayload.ID, msg.Type)
			if msg.Type == "error" {
				if errMsg, ok := msg.Payload["error"].(string); ok {
					lastError = errors.New(errMsg)
				}
			} else {
				lastError = fmt.Errorf("received non-successful status on attempt %d", i+1)
			}

		case <-ctx.Done():
			log.Printf("Gateway Timeout on attempt %d for request %s", i+1, wsPayload.ID)
			lastError = ctx.Err()
		}
		cancel()
		pendingRequests.Delete(wsPayload.ID) // 清理
	}

	// 如果循环结束，意味着所有尝试都失败了
	log.Printf("All %d connection attempts failed for request %s. Last error: %v", len(allConns), wsPayload.ID, lastError)
	recordErrorRequest(http.StatusBadGateway)
	http.Error(w, "Bad Gateway: All available client connections failed to respond successfully", http.StatusBadGateway)
}

// handleNativeGeminiProxy 处理原生的Gemini API请求
func handleNativeGeminiProxy(w http.ResponseWriter, r *http.Request) {
	log.Printf("Received native Gemini request: Method=%s, Path=%s, From=%s", r.Method, r.URL.Path, r.RemoteAddr)

	// 提取模型名称但先不记录
	var modelName string
	parts := strings.Split(r.URL.Path, "/")
	for i, part := range parts {
		if part == "models" && i+1 < len(parts) {
			modelAndAction := parts[i+1]
			modelNameParts := strings.Split(modelAndAction, ":")
			if len(modelNameParts) > 0 {
				modelName = modelNameParts[0]
			}
			break // 找到就退出
		}
	}

	// 1. 使用专门为原生Gemini请求设计的宽松认证
	userID, err := authenticateNativeRequest(r)
	if err != nil {
		recordErrorRequest(http.StatusUnauthorized)
		http.Error(w, err.Error(), http.StatusUnauthorized)
		return
	}

	// 2. 准备WebSocket消息
	reqID := uuid.NewString()
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		recordErrorRequest(http.StatusInternalServerError)
		http.Error(w, "Failed to read request body", http.StatusInternalServerError)
		return
	}
	defer r.Body.Close()

	headers := make(map[string][]string)
	for k, v := range r.Header {
		// 过滤掉一些不应转发的头
		if k != "Connection" && k != "Keep-Alive" && k != "Proxy-Authenticate" && k != "Proxy-Authorization" && k != "Te" && k != "Trailers" && k != "Transfer-Encoding" && k != "Upgrade" {
			headers[k] = v
		}
	}
	// 对于原生请求，我们不再强制覆盖API Key，让浏览器客户端自行处理

	requestPayload := WSMessage{
		ID:   reqID,
		Type: "http_request",
		Payload: map[string]interface{}{
			"method":  r.Method,
			"url":     "https://generativelanguage.googleapis.com" + r.URL.String(),
			"headers": headers,
			"body":    string(bodyBytes),
		},
	}

	// 3. 调用核心转发逻辑，并传入模型名称用于成功后记录
	forwardRequestToBrowser(w, r, userID, requestPayload, modelName)
}

// processWebSocketResponse 处理来自成功连接的后续响应
func processWebSocketResponse(w http.ResponseWriter, r *http.Request, respChan chan *WSMessage, modelName string, initialMsg *WSMessage) {
	ctx, cancel := context.WithTimeout(r.Context(), proxyRequestTimeout)
	defer cancel()

	flusher, ok := w.(http.Flusher)
	if !ok {
		log.Println("Warning: ResponseWriter does not support flushing, streaming will be buffered.")
	}

	var lastUsage *UsageMetadata // 暂存最后一个有效的 usage metadata
	headersSet := false

	// 处理已经收到的初始成功消息
	processMessage := func(msg *WSMessage) bool {
		switch msg.Type {
		case "http_response":
			if headersSet {
				log.Println("Received http_response after headers were already set. Ignoring.")
				return true // 结束
			}
			status, ok := msg.Payload["status"].(float64)
			statusCode := http.StatusOK
			if ok {
				statusCode = int(status)
			}
			if statusCode == http.StatusOK && modelName != "" {
				var usage *UsageMetadata
				if body, ok := msg.Payload["body"].(string); ok {
					var geminiResp GeminiResponse
					if err := json.Unmarshal([]byte(body), &geminiResp); err == nil {
						usage = &geminiResp.UsageMetadata
					} else {
						log.Printf("Could not unmarshal gemini response for token usage: %v", err)
					}
				}
				recordUsage(modelName, usage)
			}
			setResponseHeaders(w, msg.Payload)
			w.WriteHeader(statusCode)
			writeBody(w, msg.Payload)
			return true // 结束

		case "stream_start":
			if headersSet {
				log.Println("Received stream_start after headers were already set. Ignoring.")
				return false // 继续
			}
			setResponseHeaders(w, msg.Payload)
			writeStatusCode(w, msg.Payload)
			headersSet = true
			if flusher != nil {
				flusher.Flush()
			}

		case "stream_chunk":
			if !headersSet {
				log.Println("Warning: Received stream_chunk before stream_start. Using default 200 OK.")
				w.WriteHeader(http.StatusOK)
				headersSet = true
			}
			// 尝试解析 token usage
			if data, ok := msg.Payload["data"].(string); ok {
				bodyStr := data
				if strings.HasPrefix(bodyStr, "data: ") {
					bodyStr = strings.TrimPrefix(bodyStr, "data: ")
				}
				var geminiResp GeminiResponse
				if err := json.Unmarshal([]byte(bodyStr), &geminiResp); err == nil {
					if geminiResp.UsageMetadata.TotalTokenCount > 0 {
						lastUsage = &geminiResp.UsageMetadata
					}
				}
			}
			writeBody(w, msg.Payload)
			if flusher != nil {
				flusher.Flush()
			}

		case "stream_end":
			if modelName != "" {
				recordUsage(modelName, lastUsage) // 使用暂存的 usage
			}
			if !headersSet {
				w.WriteHeader(http.StatusOK)
			}
			return true // 结束

		case "error":
			if !headersSet {
				errMsg := "Bad Gateway: Client reported an error"
				if payloadErr, ok := msg.Payload["error"].(string); ok {
					errMsg = payloadErr
				}
				statusCode := http.StatusBadGateway
				if code, ok := msg.Payload["status"].(float64); ok {
					statusCode = int(code)
				}
				recordErrorRequest(statusCode)
				http.Error(w, errMsg, statusCode)
			} else {
				log.Printf("Error received from client after stream started: %v", msg.Payload)
			}
			return true // 结束

		default:
			log.Printf("Received unexpected message type %s while waiting for response", msg.Type)
		}
		return false // 默认继续
	}

	if processMessage(initialMsg) {
		return
	}

	// 循环处理后续消息
	for {
		select {
		case msg, ok := <-respChan:
			if !ok {
				if !headersSet {
					recordErrorRequest(http.StatusInternalServerError)
					http.Error(w, "Internal Server Error: Response channel closed unexpectedly", http.StatusInternalServerError)
				}
				return
			}
			if processMessage(msg) {
				return
			}
		case <-ctx.Done():
			if !headersSet {
				log.Printf("Gateway Timeout: No response from client for request %s", r.URL.Path)
				recordErrorRequest(http.StatusGatewayTimeout)
				http.Error(w, "Gateway Timeout", http.StatusGatewayTimeout)
			} else {
				log.Printf("Gateway Timeout: Stream incomplete for request %s", r.URL.Path)
			}
			return
		}
	}
}

// --- 辅助函数 ---

// setResponseHeaders 从payload中解析并设置HTTP响应头
func setResponseHeaders(w http.ResponseWriter, payload map[string]interface{}) {
	headers, ok := payload["headers"].(map[string]interface{})
	if !ok {
		return
	}
	for key, value := range headers {
		// 假设值是 []interface{} 或 string
		if values, ok := value.([]interface{}); ok {
			for _, v := range values {
				if strV, ok := v.(string); ok {
					w.Header().Add(key, strV)
				}
			}
		} else if strV, ok := value.(string); ok {
			w.Header().Set(key, strV)
		}
	}
}

// writeStatusCode 从payload中解析并设置HTTP状态码
func writeStatusCode(w http.ResponseWriter, payload map[string]interface{}) {
	status, ok := payload["status"].(float64) // JSON数字默认为float64
	statusCode := http.StatusOK
	if ok {
		statusCode = int(status)
	}

	if statusCode != http.StatusOK {
		recordErrorRequest(statusCode)
	}
	w.WriteHeader(statusCode)
}

// writeBody 从payload中解析并写入HTTP响应体
func writeBody(w http.ResponseWriter, payload map[string]interface{}) {
	var bodyData []byte
	// 对于 http_response，body 键通常包含数据
	if body, ok := payload["body"].(string); ok {
		bodyData = []byte(body)
	}
	// 对于 stream_chunk，data 键通常包含数据
	if data, ok := payload["data"].(string); ok {
		bodyData = []byte(data)
	}
	// 注意：如果前端发送的是二进制数据，这里应该假设它是base64编码的字符串并进行解码

	if len(bodyData) > 0 {
		w.Write(bodyData)
	}
}

// validateJWT 模拟JWT验证并返回userID
func validateJWT(token string) (string, error) {
	if token == "" {
		return "", errors.New("missing auth_token")
	}
	// 实际应用中，这里需要使用JWT库（如golang-jwt/jwt）来验证签名和过期时间
	// 这里我们简单地将token当作userID
	if token == "valid-token-user-1" {
		return "user-1", nil
	}
	//if token == "valid-token-user-2" {
	//	return "user-2", nil
	//}
	return "", errors.New("invalid token")
}

// authenticateNativeRequest 模拟HTTP代理请求的认证,不强制要求API Key
func authenticateNativeRequest(r *http.Request) (string, error) {
	// 对于原生Gemini请求，我们不在代理层面强制执行密钥检查。
	// 浏览器客户端（扩展）应该自己处理身份验证。
	// 我们只分配一个用户ID用于路由。
	return "user-1", nil
}

// authenticateAndGetGoogleKey 负责认证并返回应该用于请求Google的API密钥
func authenticateAndGetGoogleKey(r *http.Request) (userID, googleAPIKey string, err error) {
	// 1. 从所有可能的位置提取客户端提供的API Key
	clientAPIKey := r.Header.Get("x-goog-api-key")
	if clientAPIKey == "" {
		clientAPIKey = r.URL.Query().Get("key")
	}
	if clientAPIKey == "" {
		authHeader := r.Header.Get("Authorization")
		if strings.HasPrefix(authHeader, "Bearer ") {
			clientAPIKey = strings.TrimPrefix(authHeader, "Bearer ")
		}
	}

	// 2. 从环境变量中获取代理的认证密钥
	// 认证逻辑已禁用. 直接使用客户端提供的密钥.
	if clientAPIKey == "" {
		return "", "", errors.New("API key not found in request")
	}
	return "user-1", clientAPIKey, nil
}

// --- CORS 中间件 ---
func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// 设置允许跨域的响应头
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
		w.Header().Set("Access-Control-Allow-Headers", "Accept, Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization, x-goog-api-key")

		// 如果是预检请求 (OPTIONS)，则直接返回
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		// 否则，继续处理请求
		next.ServeHTTP(w, r)
	})
}

// --- Conversion Logic ---

func convertOpenAIToGemini(req *OpenAIRequest) (*GeminiRequest, error) {
	geminiContents := []GeminiContent{}
	for _, msg := range req.Messages {
		var role string
		if msg.Role == "assistant" {
			role = "model"
		} else {
			// OpenAI's "system" role is treated as a "user" role in Gemini's alternating structure
			role = "user"
		}

		var parts []GeminiPart
		switch content := msg.Content.(type) {
		case string:
			parts = append(parts, GeminiPart{Text: content})
		case []interface{}:
			for _, p := range content {
				partMap, ok := p.(map[string]interface{})
				if !ok {
					return nil, errors.New("invalid content part format")
				}
				partType, _ := partMap["type"].(string)
				if partType == "text" {
					text, _ := partMap["text"].(string)
					parts = append(parts, GeminiPart{Text: text})
				} else if partType == "image_url" {
					imageURLMap, _ := partMap["image_url"].(map[string]interface{})
					url, _ := imageURLMap["url"].(string)
					// Handle base64 encoded images
					if strings.HasPrefix(url, "data:image/") {
						parts = append(parts, GeminiPart{
							InlineData: &InlineData{
								MimeType: strings.Split(strings.Split(url, ";")[0], ":")[1],
								Data:     strings.Split(url, ",")[1],
							},
						})
					}
				}
			}
		}
		// Ensure we don't add empty content
		if len(parts) > 0 {
			geminiContents = append(geminiContents, GeminiContent{Role: role, Parts: parts})
		}
	}

	// Gemini requires the conversation to start with a "user" role.
	// If the first message is not from the user, we should probably return an error.
	if len(geminiContents) > 0 && geminiContents[0].Role != "user" {
		// Or prepend an empty user message, depending on desired behavior.
		// For now, let's assume valid input.
	}

	geminiReq := &GeminiRequest{
		Contents: geminiContents,
		GenerationConfig: &GenerationConfig{
			Temperature:     req.Temperature,
			TopP:            req.TopP,
			MaxOutputTokens: req.MaxTokens,
			StopSequences:   req.Stop,
		},
	}
	return geminiReq, nil
}

func convertGeminiToOpenAI(geminiResp *GeminiResponse, model string) *OpenAIResponse {
	choices := []OpenAIChoice{}
	for i, candidate := range geminiResp.Candidates {
		choices = append(choices, OpenAIChoice{
			Index: i,
			Message: OpenAIMessage{
				Role:    "assistant",
				Content: candidate.Content.Parts[0].Text,
			},
			FinishReason: candidate.FinishReason,
		})
	}

	return &OpenAIResponse{
		ID:      "chatcmpl-" + uuid.NewString(),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: choices,
		Usage:   geminiResp.UsageMetadata,
	}
}

func convertGeminiModelListToOpenAI(geminiResp *GeminiModelListResponse) map[string]interface{} {
	data := []map[string]interface{}{}
	for _, model := range geminiResp.Models {
		// Extract the model ID from the full name, e.g., "models/gemini-1.5-pro-latest" -> "gemini-1.5-pro-latest"
		modelID := model.Name
		if parts := strings.Split(model.Name, "/"); len(parts) == 2 {
			modelID = parts[1]
		}
		data = append(data, map[string]interface{}{
			"id":       modelID,
			"object":   "model",
			"owned_by": "google",
		})
	}

	return map[string]interface{}{
		"object": "list",
		"data":   data,
	}
}

// --- HTTP Handlers for OpenAI Compatibility ---

func handleOpenAIModels(w http.ResponseWriter, r *http.Request) {
	log.Println("Received OpenAI compatible models list request")

	// --- Authenticate the request and get the key for Google ---
	// We use the same authentication as the chat proxy, as the client should provide a key
	userID, googleAPIKey, err := authenticateAndGetGoogleKey(r)
	if err != nil {
		recordErrorRequest(http.StatusUnauthorized)
		http.Error(w, err.Error(), http.StatusUnauthorized)
		return
	}

	// --- Prepare payload for WebSocket ---
	headers := make(map[string][]string)
	for k, v := range r.Header {
		headers[k] = v
	}
	headers["x-goog-api-key"] = []string{googleAPIKey}

	wsPayload := WSMessage{
		ID:   uuid.NewString(),
		Type: "http_request",
		Payload: map[string]interface{}{
			"method":  "GET",
			"url":     "https://generativelanguage.googleapis.com/v1beta/models",
			"headers": headers,
			"body":    "", // No body for GET request
		},
	}

	// --- Use a ResponseRecorder to capture the output of the core logic ---
	recorder := httptest.NewRecorder()

	// Directly call the core logic that talks to the browser
	// modelName 传空字符串，因为这个函数会自己处理成功记录
	forwardRequestToBrowser(recorder, r, userID, wsPayload, "")

	// --- Process and convert the response back ---
	if recorder.Code != http.StatusOK {
		// 如果代理逻辑返回了非200错误，记录它
		recordErrorRequest(recorder.Code)
		w.WriteHeader(recorder.Code)
		w.Write(recorder.Body.Bytes())
		return
	}
	// 只有成功时才记录
	recordUsage("models-list", nil) // 使用一个固定的名称来记录模型列表的调用

	// Handle non-streaming response
	var geminiResp GeminiModelListResponse
	if err := json.NewDecoder(recorder.Body).Decode(&geminiResp); err != nil {
		recordErrorRequest(http.StatusInternalServerError)
		http.Error(w, "Failed to decode gemini models response: "+err.Error(), http.StatusInternalServerError)
		return
	}

	openAIResp := convertGeminiModelListToOpenAI(&geminiResp)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(openAIResp)
}

func handleOpenAIProxy(w http.ResponseWriter, r *http.Request) {
	log.Println("Received OpenAI compatible request")

	var openAIReq OpenAIRequest
	if err := json.NewDecoder(r.Body).Decode(&openAIReq); err != nil {
		recordErrorRequest(http.StatusBadRequest)
		http.Error(w, "Invalid request body: "+err.Error(), http.StatusBadRequest)
		return
	}

	geminiReq, err := convertOpenAIToGemini(&openAIReq)
	if err != nil {
		recordErrorRequest(http.StatusInternalServerError)
		http.Error(w, "Failed to convert request: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// --- Authenticate the request and get the key for Google ---
	userID, googleAPIKey, err := authenticateAndGetGoogleKey(r)
	if err != nil {
		recordErrorRequest(http.StatusUnauthorized)
		http.Error(w, err.Error(), http.StatusUnauthorized)
		return
	}

	// --- Prepare payload for WebSocket ---
	geminiReqBytes, err := json.Marshal(geminiReq)
	if err != nil {
		recordErrorRequest(http.StatusInternalServerError)
		http.Error(w, "Failed to marshal gemini request", http.StatusInternalServerError)
		return
	}

	// The URL in the payload for the browser does not depend on the original request URL,
	// so we can construct a generic one.
	modelToUse := openAIReq.Model
	if modelToUse == "" {
		modelToUse = "gemini-1.5-pro-latest" // Default model
	}
	action := "generateContent"
	if openAIReq.Stream {
		action = "streamGenerateContent"
	}

	// Forward all headers from the original request, but ensure the correct API key is set
	headers := make(map[string][]string)
	for k, v := range r.Header {
		headers[k] = v
	}
	headers["x-goog-api-key"] = []string{googleAPIKey}

	wsPayload := WSMessage{
		ID:   uuid.NewString(),
		Type: "http_request",
		Payload: map[string]interface{}{
			"method":  "POST",
			"url":     "https://generativelanguage.googleapis.com/v1beta/models/" + modelToUse + ":" + action,
			"headers": headers,
			"body":    string(geminiReqBytes),
		},
	}

	if openAIReq.Stream {
		// For streaming, we handle the response asynchronously without a recorder.
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		flusher, ok := w.(http.Flusher)
		if !ok {
			recordErrorRequest(http.StatusInternalServerError)
			http.Error(w, "Streaming unsupported!", http.StatusInternalServerError)
			return
		}

		// This part is similar to forwardRequestToBrowser, but we handle the response asynchronously.
		respChan := make(chan *WSMessage, 10)
		pendingRequests.Store(wsPayload.ID, respChan)
		defer pendingRequests.Delete(wsPayload.ID)

		selectedConn, err := globalPool.GetConnection(userID)
		if err != nil {
			recordErrorRequest(http.StatusServiceUnavailable)
			http.Error(w, "Service Unavailable: No active client connected", http.StatusServiceUnavailable)
			return
		}

		if err := selectedConn.safeWriteJSON(wsPayload); err != nil {
			recordErrorRequest(http.StatusBadGateway)
			http.Error(w, "Bad Gateway: Failed to send request to client", http.StatusBadGateway)
			return
		}

		// Asynchronous response processing loop.
		ctx, cancel := context.WithTimeout(r.Context(), proxyRequestTimeout)
		defer cancel()

		var lastUsage *UsageMetadata // 在循环外定义

		for {
			select {
			case msg, ok := <-respChan:
				if !ok {
					// Channel closed, stream is done.
					fmt.Fprintf(w, "data: [DONE]\n\n")
					flusher.Flush()
					return
				}

				switch msg.Type {
				case "stream_chunk":
					var bodyStr string
					// The raw JSON data from Gemini is in the 'data' field of the payload.
					if data, ok := msg.Payload["data"].(string); ok {
						bodyStr = data
					} else {
						continue
					}

					// The browser client might forward the raw SSE line, including "data: ".
					// We need to trim it before parsing.
					if strings.HasPrefix(bodyStr, "data: ") {
						bodyStr = strings.TrimPrefix(bodyStr, "data: ")
					}

					var geminiResp GeminiResponse
					if err := json.Unmarshal([]byte(bodyStr), &geminiResp); err != nil {
						log.Printf("Error unmarshalling stream chunk from WS payload: %v", err)
						continue
					}

					// 暂存 usage metadata
					if geminiResp.UsageMetadata.TotalTokenCount > 0 {
						lastUsage = &geminiResp.UsageMetadata
					}

					if len(geminiResp.Candidates) > 0 && len(geminiResp.Candidates[0].Content.Parts) > 0 {
						openAIChunk := OpenAIStreamResponse{
							ID:      "chatcmpl-" + uuid.NewString(),
							Object:  "chat.completion.chunk",
							Created: time.Now().Unix(),
							Model:   modelToUse,
							Choices: []OpenAIStreamChoice{
								{
									Index: 0,
									Delta: OpenAIMessage{
										Role:    "assistant",
										Content: geminiResp.Candidates[0].Content.Parts[0].Text,
									},
								},
							},
						}
						chunkBytes, err := json.Marshal(openAIChunk)
						if err != nil {
							log.Printf("Error marshalling OpenAI chunk: %v", err)
							continue
						}
						fmt.Fprintf(w, "data: %s\n\n", chunkBytes)
						flusher.Flush()
					}

				case "stream_end":
					recordUsage(openAIReq.Model, lastUsage) // 使用暂存的 usage
					fmt.Fprintf(w, "data: [DONE]\n\n")
					flusher.Flush()
					return

				case "error":
					log.Printf("Received error from client during stream: %v", msg.Payload)
					// We can't send an HTTP error now, so we just terminate the stream.
					fmt.Fprintf(w, "data: [DONE]\n\n")
					flusher.Flush()
					return
				}

			case <-ctx.Done():
				log.Printf("Gateway Timeout: Stream incomplete for request %s", r.URL.Path)
				// Terminate the stream on timeout.
				fmt.Fprintf(w, "data: [DONE]\n\n")
				flusher.Flush()
				return
			}
		}
	} else {
		// For non-streaming, the original recorder logic is correct.
		recorder := httptest.NewRecorder()
		// modelName 传空字符串，让此函数自己处理成功记录
		forwardRequestToBrowser(recorder, r, userID, wsPayload, "")

		if recorder.Code != http.StatusOK {
			recordErrorRequest(recorder.Code)
			w.WriteHeader(recorder.Code)
			w.Write(recorder.Body.Bytes())
			return
		}

		// 只有成功时才记录
		var geminiResp GeminiResponse
		if err := json.NewDecoder(recorder.Body).Decode(&geminiResp); err != nil {
			recordErrorRequest(http.StatusInternalServerError)
			http.Error(w, "Failed to decode gemini response: "+err.Error(), http.StatusInternalServerError)
			return
		}
		recordUsage(openAIReq.Model, &geminiResp.UsageMetadata)

		openAIResp := convertGeminiToOpenAI(&geminiResp, openAIReq.Model)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(openAIResp)
	}
}

// --- API统计与前端展示 ---

var statsHTML []byte // 用于缓存 status.html 的内容

// handleStats 以JSON格式返回当前的API使用统计数据
func handleStats(w http.ResponseWriter, r *http.Request) {
	daysStr := r.URL.Query().Get("days")
	days, err := strconv.Atoi(daysStr)
	if err != nil || days == 0 {
		// A value of 0 or an invalid value means all time
		days = 0
	}

	query := "SELECT model_name, date, count, prompt_tokens, candidates_tokens, total_tokens, cost FROM usage_stats"
	var args []interface{}

	if days > 0 {
		// We want to include `days` number of days. e.g., if days=7, we want today + 6 previous days.
		cutoffDate := time.Now().In(cstZone).AddDate(0, 0, -(days - 1)).Format("2006-01-02")
		query += " WHERE date >= ?"
		args = append(args, cutoffDate)
	}
	query += " ORDER BY date, model_name"

	rows, err := db.Query(query, args...)
	if err != nil {
		log.Printf("从数据库查询统计数据时出错: %v", err)
		http.Error(w, "无法查询统计数据", http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	statsCopy := make(map[string]UsageData)
	for rows.Next() {
		var modelName, date string
		var data UsageData
		if err := rows.Scan(&modelName, &date, &data.Count, &data.PromptTokens, &data.CandidatesTokens, &data.TotalTokens, &data.Cost); err != nil {
			log.Printf("扫描数据库行时出错: %v", err)
			continue // 跳过此行
		}
		// 为了与前端兼容，重新组合成旧的key格式
		key := fmt.Sprintf("%s-%s", modelName, date)
		statsCopy[key] = data
	}

	if err := rows.Err(); err != nil {
		log.Printf("遍历数据库行时出错: %v", err)
		http.Error(w, "遍历统计数据时出错", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(statsCopy); err != nil {
		log.Printf("序列化统计数据时出错: %v", err)
		http.Error(w, "无法序列化统计数据", http.StatusInternalServerError)
	}
}

// handleConnections returns the current WebSocket connection status.
func handleConnections(w http.ResponseWriter, r *http.Request) {
	globalPool.RLock()
	defer globalPool.RUnlock()

	status := make(map[string]UserConnectionStatus)

	for userID, userConns := range globalPool.Users {
		userConns.Lock() // Lock individual user's connections

		connInfos := make([]ConnectionInfo, 0, len(userConns.Connections))
		for _, conn := range userConns.Connections {
			connInfos = append(connInfos, ConnectionInfo{
				RemoteAddr: conn.Conn.RemoteAddr().String(),
				LastActive: conn.LastActive,
				Healthy:    conn.Healthy,
			})
		}

		status[userID] = UserConnectionStatus{
			ConnectionCount: len(userConns.Connections),
			Connections:     connInfos,
		}

		userConns.Unlock()
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(status); err != nil {
		log.Printf("序列化连接状态时出错: %v", err)
		http.Error(w, "无法序列化连接状态", http.StatusInternalServerError)
	}
}

// DailyCostStat holds cost per model for a specific day.
type DailyCostStat struct {
	ModelName string  `json:"model_name"`
	Cost      float64 `json:"cost"`
}

// handleTodayCostStats 以JSON格式返回当天各模型的费用消耗
func handleTodayCostStats(w http.ResponseWriter, r *http.Request) {
	date := time.Now().In(cstZone).Format("2006-01-02")
	rows, err := db.Query("SELECT model_name, cost FROM usage_stats WHERE date = ? AND cost > 0 ORDER BY cost DESC", date)
	if err != nil {
		log.Printf("从数据库查询今日费用统计数据时出错: %v", err)
		http.Error(w, "无法查询今日费用统计数据", http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	var stats []DailyCostStat
	for rows.Next() {
		var stat DailyCostStat
		if err := rows.Scan(&stat.ModelName, &stat.Cost); err != nil {
			log.Printf("扫描数据库行时出错: %v", err)
			continue
		}
		stats = append(stats, stat)
	}

	if err := rows.Err(); err != nil {
		log.Printf("遍历数据库行时出错: %v", err)
		http.Error(w, "遍历今日费用统计数据时出错", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(stats); err != nil {
		log.Printf("序列化今日费用统计数据时出错: %v", err)
		http.Error(w, "无法序列化今日费用统计数据", http.StatusInternalServerError)
	}
}

// --- 主函数 ---

func main() {
	// 在程序启动时加载一次HTML文件
	var err error
	statsHTML, err = os.ReadFile("status.html")
	if err != nil {
		log.Fatalf("无法读取 status.html: %v", err)
	}

	// 初始化数据库
	initDB()
	defer db.Close()

	// 启动一个goroutine来定期恢复不健康的连接
	go func() {
		ticker := time.NewTicker(30 * time.Second) // 每30秒检查一次
		defer ticker.Stop()
		for range ticker.C {
			recoverUnhealthyConnections()
		}
	}()

	// 创建一个新的 ServeMux
	mux := http.NewServeMux()

	// WebSocket 路由 (通常不需要CORS)
	mux.HandleFunc(wsPath, handleWebSocket)

	// HTTP 反向代理路由
	mux.HandleFunc("/v1beta/", handleNativeGeminiProxy)
	mux.HandleFunc("/v1/", handleNativeGeminiProxy) // 兼容v1

	// OpenAI 兼容路由
	mux.HandleFunc("/v1/chat/completions", handleOpenAIProxy)
	mux.HandleFunc("/v1/models", handleOpenAIModels)

	// API 统计路由
	mux.HandleFunc("/stats", handleStats)
	mux.HandleFunc("/connections", handleConnections)
	mux.HandleFunc("/stats/today_cost", handleTodayCostStats)
	// 根路由和静态文件服务
	// 创建一个文件服务器，为当前目录下的文件提供服务
	fs := http.FileServer(http.Dir("."))
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		// 如果请求的是根路径，我们专门提供 status.html
		if r.URL.Path == "/" {
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.Write(statsHTML)
			return
		}
		// 对于所有其他路径 (例如 /echarts.min.js), 让文件服务器来处理
		fs.ServeHTTP(w, r)
	})

	// 将 CORS 中间件应用到所有 HTTP 路由
	handler := corsMiddleware(mux)

	log.Printf("Starting server on %s", proxyListenAddr)
	log.Printf("WebSocket endpoint available at ws://%s%s", proxyListenAddr, wsPath)
	log.Printf("HTTP proxy available at http://%s/", proxyListenAddr)

	if err := http.ListenAndServe(proxyListenAddr, handler); err != nil {
		log.Fatalf("Could not start server: %s\n", err)
	}
}
