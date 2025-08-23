package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
)

// --- Constants ---
const (
	wsPath              = "/v1/ws"
	proxyListenAddr     = ":5345"
	wsReadTimeout       = 60 * time.Second
	proxyRequestTimeout = 600 * time.Second
)

// --- API Usage Statistics ---

const statsFilename = "usage_stats.json"

var (
	// key: "modelName-YYYY-MM-DD", value: count
	usageStats      = make(map[string]int64)
	usageStatsMutex sync.RWMutex
)

// loadStats 从文件中加载统计数据
func loadStats() {
	usageStatsMutex.Lock()
	defer usageStatsMutex.Unlock()

	data, err := os.ReadFile(statsFilename)
	if err != nil {
		if os.IsNotExist(err) {
			log.Println("统计文件不存在, 将创建一个新的。")
			usageStats = make(map[string]int64)
		} else {
			log.Printf("加载统计文件时出错: %v", err)
		}
		return
	}

	if err := json.Unmarshal(data, &usageStats); err != nil {
		log.Printf("解析统计文件时出错: %v", err)
		// 如果文件损坏，则从一个空的map开始
		usageStats = make(map[string]int64)
	}
	log.Printf("成功从 %s 加载了 %d 条统计记录。", statsFilename, len(usageStats))
}

// saveStats 将统计数据保存到文件
func saveStats() {
	usageStatsMutex.RLock()
	// 复制map以最小化锁的持有时间
	statsCopy := make(map[string]int64, len(usageStats))
	for k, v := range usageStats {
		statsCopy[k] = v
	}
	usageStatsMutex.RUnlock()

	data, err := json.MarshalIndent(statsCopy, "", "  ")
	if err != nil {
		log.Printf("序列化统计数据时出错: %v", err)
		return
	}

	// 原子写入：先写入临时文件，然后重命名
	tempFilename := statsFilename + ".tmp"
	if err := os.WriteFile(tempFilename, data, 0644); err != nil {
		log.Printf("写入临时统计文件时出错: %v", err)
		return
	}
	if err := os.Rename(tempFilename, statsFilename); err != nil {
		log.Printf("重命名统计文件时出错: %v", err)
	}
	log.Printf("已将统计数据持久化到 %s", statsFilename)
}

// recordUsage increments the usage count for a given model.
func recordUsage(modelName string) {
	// Sanitize model name to avoid issues with path separators etc.
	// e.g. "gemini-1.5-pro-latest"
	if modelName == "" {
		return
	}
	date := time.Now().UTC().Format("2006-01-02")
	key := fmt.Sprintf("%s-%s", modelName, date)

	usageStatsMutex.Lock()
	defer usageStatsMutex.Unlock()

	usageStats[key]++
	log.Printf("Recorded usage for model %s. Total for today: %d", modelName, usageStats[key])
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

// --- 1. 连接管理与负载均衡 ---

// UserConnection 存储单个WebSocket连接及其元数据
type UserConnection struct {
	Conn       *websocket.Conn
	UserID     string
	LastActive time.Time
	writeMutex sync.Mutex // 保护对此单个连接的并发写入
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
func forwardRequestToBrowser(w http.ResponseWriter, r *http.Request, userID string, wsPayload WSMessage) {
	// 1. 创建响应通道并注册
	respChan := make(chan *WSMessage, 10)
	pendingRequests.Store(wsPayload.ID, respChan)
	defer pendingRequests.Delete(wsPayload.ID)

	// 2. 选择一个WebSocket连接
	selectedConn, err := globalPool.GetConnection(userID)
	if err != nil {
		log.Printf("Error getting connection for user %s: %v", userID, err)
		http.Error(w, "Service Unavailable: No active client connected", http.StatusServiceUnavailable)
		return
	}

	// 3. 发送请求到WebSocket客户端
	if err := selectedConn.safeWriteJSON(wsPayload); err != nil {
		log.Printf("Failed to send request over WebSocket: %v", err)
		http.Error(w, "Bad Gateway: Failed to send request to client", http.StatusBadGateway)
		return
	}

	// 4. 异步等待并处理响应
	processWebSocketResponse(w, r, respChan)
}

// handleNativeGeminiProxy 处理原生的Gemini API请求
func handleNativeGeminiProxy(w http.ResponseWriter, r *http.Request) {
	log.Printf("Received native Gemini request: Method=%s, Path=%s, From=%s", r.Method, r.URL.Path, r.RemoteAddr)

	// 提取模型名称并记录使用情况
	go func(path string) {
		parts := strings.Split(path, "/")
		for i, part := range parts {
			if part == "models" && i+1 < len(parts) {
				modelAndAction := parts[i+1]
				modelNameParts := strings.Split(modelAndAction, ":")
				if len(modelNameParts) > 0 {
					recordUsage(modelNameParts[0])
				}
				return
			}
		}
	}(r.URL.Path)

	// 1. 使用专门为原生Gemini请求设计的宽松认证
	userID, err := authenticateNativeRequest(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusUnauthorized)
		return
	}

	// 2. 准备WebSocket消息
	reqID := uuid.NewString()
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
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

	// 3. 调用核心转发逻辑
	forwardRequestToBrowser(w, r, userID, requestPayload)
}

// processWebSocketResponse 处理来自WS通道的响应，构建HTTP响应
func processWebSocketResponse(w http.ResponseWriter, r *http.Request, respChan chan *WSMessage) {
	// 设置超时
	ctx, cancel := context.WithTimeout(r.Context(), proxyRequestTimeout)
	defer cancel()

	// 获取Flusher以支持流式响应
	flusher, ok := w.(http.Flusher)
	if !ok {
		log.Println("Warning: ResponseWriter does not support flushing, streaming will be buffered.")
	}

	headersSet := false

	for {
		select {
		case msg, ok := <-respChan:
			if !ok {
				// 通道被关闭，理论上不应该发生，除非有panic
				if !headersSet {
					http.Error(w, "Internal Server Error: Response channel closed unexpectedly", http.StatusInternalServerError)
				}
				return
			}

			switch msg.Type {
			case "http_response":
				// 标准单个响应
				if headersSet {
					log.Println("Received http_response after headers were already set. Ignoring.")
					return
				}
				setResponseHeaders(w, msg.Payload)
				writeStatusCode(w, msg.Payload)
				writeBody(w, msg.Payload)
				return // 请求结束

			case "stream_start":
				// 流开始
				if headersSet {
					log.Println("Received stream_start after headers were already set. Ignoring.")
					continue
				}
				setResponseHeaders(w, msg.Payload)
				writeStatusCode(w, msg.Payload)
				headersSet = true
				if flusher != nil {
					flusher.Flush()
				}

			case "stream_chunk":
				// 流数据块
				if !headersSet {
					// 如果还没收到stream_start，先设置默认头
					log.Println("Warning: Received stream_chunk before stream_start. Using default 200 OK.")
					w.WriteHeader(http.StatusOK)
					headersSet = true
				}
				writeBody(w, msg.Payload)
				if flusher != nil {
					flusher.Flush() // 立即将数据块发送给客户端
				}

			case "stream_end":
				// 流结束
				if !headersSet {
					// 如果流结束了但还没设置头，设置一个默认的
					w.WriteHeader(http.StatusOK)
				}
				return // 请求结束

			case "error":
				// 前端返回错误
				if !headersSet {
					errMsg := "Bad Gateway: Client reported an error"
					if payloadErr, ok := msg.Payload["error"].(string); ok {
						errMsg = payloadErr
					}
					statusCode := http.StatusBadGateway
					if code, ok := msg.Payload["status"].(float64); ok {
						statusCode = int(code)
					}
					http.Error(w, errMsg, statusCode)
				} else {
					// 如果已经开始发送流，我们只能记录错误并关闭连接
					log.Printf("Error received from client after stream started: %v", msg.Payload)
				}
				return // 请求结束

			default:
				log.Printf("Received unexpected message type %s while waiting for response", msg.Type)
			}

		case <-ctx.Done():
			// 超时
			if !headersSet {
				log.Printf("Gateway Timeout: No response from client for request %s", r.URL.Path)
				http.Error(w, "Gateway Timeout", http.StatusGatewayTimeout)
			} else {
				// 如果流已经开始，我们只能记录日志并断开连接
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
	if !ok {
		w.WriteHeader(http.StatusOK) // 默认200
		return
	}
	w.WriteHeader(int(status))
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
	forwardRequestToBrowser(recorder, r, userID, wsPayload)

	// --- Process and convert the response back ---
	if recorder.Code != http.StatusOK {
		// If the proxy logic returned an error, forward it
		w.WriteHeader(recorder.Code)
		w.Write(recorder.Body.Bytes())
		return
	}

	// Handle non-streaming response
	var geminiResp GeminiModelListResponse
	if err := json.NewDecoder(recorder.Body).Decode(&geminiResp); err != nil {
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
		http.Error(w, "Invalid request body: "+err.Error(), http.StatusBadRequest)
		return
	}

	// 记录模型使用情况
	go recordUsage(openAIReq.Model)

	geminiReq, err := convertOpenAIToGemini(&openAIReq)
	if err != nil {
		http.Error(w, "Failed to convert request: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// --- Authenticate the request and get the key for Google ---
	userID, googleAPIKey, err := authenticateAndGetGoogleKey(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusUnauthorized)
		return
	}

	// --- Prepare payload for WebSocket ---
	geminiReqBytes, err := json.Marshal(geminiReq)
	if err != nil {
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
			http.Error(w, "Streaming unsupported!", http.StatusInternalServerError)
			return
		}

		// This part is similar to forwardRequestToBrowser, but we handle the response asynchronously.
		respChan := make(chan *WSMessage, 10)
		pendingRequests.Store(wsPayload.ID, respChan)
		defer pendingRequests.Delete(wsPayload.ID)

		selectedConn, err := globalPool.GetConnection(userID)
		if err != nil {
			http.Error(w, "Service Unavailable: No active client connected", http.StatusServiceUnavailable)
			return
		}

		if err := selectedConn.safeWriteJSON(wsPayload); err != nil {
			http.Error(w, "Bad Gateway: Failed to send request to client", http.StatusBadGateway)
			return
		}

		// Asynchronous response processing loop.
		ctx, cancel := context.WithTimeout(r.Context(), proxyRequestTimeout)
		defer cancel()

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
		forwardRequestToBrowser(recorder, r, userID, wsPayload)

		if recorder.Code != http.StatusOK {
			w.WriteHeader(recorder.Code)
			w.Write(recorder.Body.Bytes())
			return
		}

		var geminiResp GeminiResponse
		if err := json.NewDecoder(recorder.Body).Decode(&geminiResp); err != nil {
			http.Error(w, "Failed to decode gemini response: "+err.Error(), http.StatusInternalServerError)
			return
		}

		openAIResp := convertGeminiToOpenAI(&geminiResp, openAIReq.Model)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(openAIResp)
	}
}

// --- API统计与前端展示 ---

var statsHTML []byte // 用于缓存 status.html 的内容

// handleStats 以JSON格式返回当前的API使用统计数据
func handleStats(w http.ResponseWriter, r *http.Request) {
	usageStatsMutex.RLock()
	defer usageStatsMutex.RUnlock()

	// 为了避免在JSON编码期间长时间持有锁，我们复制map
	statsCopy := make(map[string]int64, len(usageStats))
	for k, v := range usageStats {
		statsCopy[k] = v
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(statsCopy); err != nil {
		log.Printf("序列化统计数据时出错: %v", err)
		http.Error(w, "无法序列化统计数据", http.StatusInternalServerError)
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

	// 加载持久化的统计数据
	loadStats()

	// 启动一个goroutine来定期保存统计数据
	go func() {
		ticker := time.NewTicker(1 * time.Minute) // 每分钟保存一次
		defer ticker.Stop()
		for range ticker.C {
			saveStats()
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
