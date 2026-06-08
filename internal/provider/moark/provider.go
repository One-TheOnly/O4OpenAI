package moark

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/o4openai/internal/model"
	"github.com/o4openai/internal/provider"
	"github.com/o4openai/pkg/utils"
	"go.uber.org/zap"
)

// ============================================================
// Moark (模力方舟) Provider - implements model.Provider interface
//
// The Moark OpenAPI is largely OpenAI-compatible, so the conversion
// surface is intentionally small. Most endpoints accept the OpenAI
// request/response shape verbatim; the only place we do meaningful
// work is:
//   - image edits: convert base64/URL → multipart/form-data
//   - video: submit to /v1/async/videos/generations, then poll
//            /v1/task/{task_id} on subsequent retrieve calls
// ============================================================

const (
	DefaultBaseURL = "https://api.moark.com/v1"
	ProviderName   = "moark"
)

// Provider implements model.Provider for Moark.
type Provider struct {
	baseURL       string
	apiKey        string
	httpClient    *http.Client
	logger        *zap.Logger
	base64Handler *utils.Base64Handler

	// Model mappings: external model name -> Moark model name.
	modelMappings map[string]string
}

// NewProvider creates a new Moark provider.
func NewProvider(apiKey, baseURL string, logger *zap.Logger) *Provider {
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	return &Provider{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		httpClient: &http.Client{
			Timeout: 600 * time.Second, // 10 min for video generation
		},
		logger:        logger,
		modelMappings: make(map[string]string),
	}
}

// SetModelMappings configures the model name mappings.
func (p *Provider) SetModelMappings(mappings map[string]string) {
	p.modelMappings = mappings
}

// SetBase64Handler sets the base64-to-URL conversion handler. Used to
// convert base64 image data URIs that clients send into temp URLs that
// the gateway can serve, so Moark (which prefers URLs for image inputs
// in chat messages) can fetch them.
func (p *Provider) SetBase64Handler(handler *utils.Base64Handler) {
	p.base64Handler = handler
}

// Name returns the provider name.
func (p *Provider) Name() string { return ProviderName }

// SupportedModels returns the models this provider supports.
func (p *Provider) SupportedModels() []model.ModelInfo {
	// A representative slice of Moark models. Custom mappings may add
	// more at runtime; de-duplicated by ID.
	baseModels := []model.ModelInfo{
		// --- Chat / text ---
		{ID: "DeepSeek-V3_1-Terminus", Object: "model", Created: 1710000000, OwnedBy: "moark"},
		{ID: "DeepSeek-V3", Object: "model", Created: 1709000000, OwnedBy: "moark"},
		{ID: "DeepSeek-R1", Object: "model", Created: 1708000000, OwnedBy: "moark"},
		{ID: "Qwen2.5-72B-Instruct", Object: "model", Created: 1707000000, OwnedBy: "moark"},
		{ID: "Qwen3-235B-A22B-Instruct-2507", Object: "model", Created: 1750000000, OwnedBy: "moark"},
		{ID: "GLM-4.6", Object: "model", Created: 1749000000, OwnedBy: "moark"},
		{ID: "GLM-4.7", Object: "model", Created: 1751000000, OwnedBy: "moark"},
		{ID: "GLM-4.7-Flash", Object: "model", Created: 1751000000, OwnedBy: "moark"},
		{ID: "ERNIE-4.5-Turbo", Object: "model", Created: 1748000000, OwnedBy: "moark"},
		{ID: "Kimi-K2.5", Object: "model", Created: 1752000000, OwnedBy: "moark"},
		{ID: "Kimi-K2-Thinking", Object: "model", Created: 1753000000, OwnedBy: "moark"},
		// --- Image generation ---
		{ID: "FLUX.1-dev", Object: "model", Created: 1720000000, OwnedBy: "moark"},
		{ID: "Qwen-Image", Object: "model", Created: 1745000000, OwnedBy: "moark"},
		{ID: "Kolors", Object: "model", Created: 1718000000, OwnedBy: "moark"},
		{ID: "LongCat-Image", Object: "model", Created: 1746000000, OwnedBy: "moark"},
		// --- Image edit ---
		{ID: "LongCat-Image-Edit", Object: "model", Created: 1746000000, OwnedBy: "moark"},
		{ID: "Qwen-Image-Edit", Object: "model", Created: 1747000000, OwnedBy: "moark"},
		{ID: "FLUX.1-Kontext-dev", Object: "model", Created: 1742000000, OwnedBy: "moark"},
		// --- Video ---
		{ID: "Wan2.1-T2V-14B", Object: "model", Created: 1730000000, OwnedBy: "moark"},
		{ID: "Wan2.7", Object: "model", Created: 1740000000, OwnedBy: "moark"},
		{ID: "HunyuanVideo-1.5", Object: "model", Created: 1748000000, OwnedBy: "moark"},
		{ID: "CogVideoX-5b", Object: "model", Created: 1715000000, OwnedBy: "moark"},
		{ID: "ViduQ2-Turbo", Object: "model", Created: 1742000000, OwnedBy: "moark"},
		{ID: "ViduQ2-Pro", Object: "model", Created: 1743000000, OwnedBy: "moark"},
		{ID: "ViduQ3-Turbo", Object: "model", Created: 1751000000, OwnedBy: "moark"},
		{ID: "ViduQ3-Pro", Object: "model", Created: 1752000000, OwnedBy: "moark"},
		{ID: "HappyHorse-1.0", Object: "model", Created: 1749000000, OwnedBy: "moark"},
	}

	seen := make(map[string]bool, len(baseModels))
	for _, m := range baseModels {
		seen[m.ID] = true
	}
	// Add any custom mapped models
	for external := range p.modelMappings {
		if !seen[external] {
			baseModels = append(baseModels, model.ModelInfo{
				ID:      external,
				Object:  "model",
				Created: time.Now().Unix(),
				OwnedBy: "moark",
			})
		}
	}
	return baseModels
}

// Capability reports.
func (p *Provider) SupportsChat() bool            { return true }
func (p *Provider) SupportsImageGeneration() bool { return true }
func (p *Provider) SupportsImageEdit() bool       { return true }
func (p *Provider) SupportsImageVariation() bool  { return false } // no Moark endpoint
func (p *Provider) SupportsVideoGeneration() bool { return true }

// FetchModels calls GET /v1/models on the Moark API and returns the list
// of model IDs the upstream currently exposes. Used at startup when the
// user hasn't pinned a model list in config.yaml.
func (p *Provider) FetchModels(ctx context.Context) ([]string, error) {
	apiKey := p.apiKeyFromContext(ctx)
	if apiKey == "" {
		apiKey = p.apiKey
	}
	if apiKey == "" {
		return nil, fmt.Errorf("moark: cannot fetch models without an API key")
	}

	req, err := http.NewRequestWithContext(ctx, "GET", p.baseURL+"/models", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create models request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)

	p.logger.Info("Fetching live model list from Moark", zap.String("url", p.baseURL+"/models"))
	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("models request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return nil, &provider.ProviderError{
			StatusCode: resp.StatusCode,
			Body:       string(body),
			Provider:   "Moark",
		}
	}

	var list struct {
		Object string `json:"object"`
		Data   []struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			Created int64  `json:"created"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		return nil, fmt.Errorf("failed to decode models response: %w", err)
	}

	ids := make([]string, 0, len(list.Data))
	for _, m := range list.Data {
		if m.ID != "" {
			ids = append(ids, m.ID)
		}
	}
	p.logger.Info("Fetched Moark model list", zap.Int("count", len(ids)))
	return ids, nil
}

// resolveModel maps an external model name to the upstream Moark model name.
func (p *Provider) resolveModel(externalModel string) string {
	if mapped, ok := p.modelMappings[externalModel]; ok {
		return mapped
	}
	return externalModel
}

// apiKeyFromContext returns the client-supplied API key, falling back to
// the provider's configured key when available.
func (p *Provider) apiKeyFromContext(ctx context.Context) string {
	if k := utils.APIKeyFromCtx(ctx); k != "" {
		return k
	}
	return p.apiKey
}

// doJSON performs a JSON HTTP request to the Moark API and returns the
// response. The caller is responsible for closing resp.Body.
func (p *Provider) doJSON(ctx context.Context, method, path string, body interface{}) (*http.Response, error) {
	var reqBody io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request body: %w", err)
		}
		reqBody = bytes.NewReader(data)
		p.logger.Info("Moark API request body",
			zap.String("path", path),
			zap.String("body", string(data)),
		)
	}
	return p.doRaw(ctx, method, path, "application/json", reqBody)
}

// doRaw performs an HTTP request with an arbitrary content type and body.
func (p *Provider) doRaw(ctx context.Context, method, path, contentType string, body io.Reader) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, method, p.baseURL+path, body)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	apiKey := p.apiKeyFromContext(ctx)
	if apiKey == "" {
		return nil, provider.ErrNoAPIKey
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}

	p.logger.Info("Moark API request",
		zap.String("method", method),
		zap.String("url", p.baseURL+path),
	)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	// Error responses — log full body without truncation.
	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		bodyBytes, _ := io.ReadAll(resp.Body)
		p.logger.Error("Moark API error response",
			zap.Int("status", resp.StatusCode),
			zap.String("body", string(bodyBytes)),
		)
		return nil, &provider.ProviderError{
			StatusCode: resp.StatusCode,
			Body:       string(bodyBytes),
			Provider:   "Moark",
		}
	}

	// Streaming (SSE) responses — don't buffer the body, just log metadata.
	respCT := resp.Header.Get("Content-Type")
	if strings.Contains(respCT, "text/event-stream") {
		p.logger.Info("Moark API response (stream)",
			zap.Int("status", resp.StatusCode),
			zap.String("content_type", respCT),
		)
		return resp, nil
	}

	// Non-streaming success — read body for logging, then restore it so
	// callers can still decode.
	bodyBytes, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}
	p.logger.Info("Moark API response",
		zap.Int("status", resp.StatusCode),
		zap.String("body", string(bodyBytes)),
	)
	resp.Body = io.NopCloser(bytes.NewReader(bodyBytes))
	return resp, nil
}

// decodeError attempts to extract a structured error message from a Moark
// error body. Falls back to the raw body when the body isn't JSON.
func (p *Provider) decodeError(status int, body string) string {
	var er MoarkErrorResponse
	if err := json.Unmarshal([]byte(body), &er); err == nil && er.Error.Message != "" {
		return er.Error.Message
	}
	return strings.TrimSpace(body)
}
