package agnes

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
// Agnes AI Provider - implements model.Provider interface
// Converts between OpenAI format and Agnes AI native format
// ============================================================

const (
	DefaultBaseURL = "https://apihub.agnes-ai.com/v1"
	ProviderName   = "agnes"
)

// Provider implements model.Provider for Agnes AI
type Provider struct {
	baseURL       string
	apiKey        string
	httpClient    *http.Client
	logger        *zap.Logger
	base64Handler *utils.Base64Handler

	// Model mappings: external model name -> Agnes model name
	modelMappings map[string]string
}

// NewProvider creates a new Agnes AI provider
func NewProvider(apiKey, baseURL string, logger *zap.Logger) *Provider {
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	return &Provider{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		httpClient: &http.Client{
			Timeout: 300 * time.Second, // 5 min timeout for video generation
		},
		logger:        logger,
		modelMappings: make(map[string]string),
	}
}

// SetModelMappings configures the model name mappings
func (p *Provider) SetModelMappings(mappings map[string]string) {
	p.modelMappings = mappings
}

// SetBase64Handler sets the base64-to-URL conversion handler.
// This is essential for converting base64 image data URIs that OpenAI clients
// send into temporary URLs that Agnes AI can fetch.
func (p *Provider) SetBase64Handler(handler *utils.Base64Handler) {
	p.base64Handler = handler
}

// Name returns the provider name
func (p *Provider) Name() string {
	return ProviderName
}

// SupportedModels returns the models this provider supports
func (p *Provider) SupportedModels() []model.ModelInfo {
	// Base models known for Agnes AI
	baseModels := []model.ModelInfo{
		{ID: "agnes-1.5-flash", Object: "model", Created: 1700000000, OwnedBy: "agnes"},
		{ID: "agnes-2.0-flash", Object: "model", Created: 1710000000, OwnedBy: "agnes"},
		{ID: "agnes-image-2.0-flash", Object: "model", Created: 1715000000, OwnedBy: "agnes"},
		{ID: "agnes-image-2.1-flash", Object: "model", Created: 1720000000, OwnedBy: "agnes"},
		{ID: "agnes-video-v2.0", Object: "model", Created: 1725000000, OwnedBy: "agnes"},
	}

	// Build a set of known model IDs
	seen := make(map[string]bool)
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
				OwnedBy: "agnes",
			})
		}
	}
	return baseModels
}

// SupportsChat reports chat capability
func (p *Provider) SupportsChat() bool { return true }

// SupportsImageGeneration reports image generation capability
func (p *Provider) SupportsImageGeneration() bool { return true }

// SupportsImageEdit reports image edit capability
// Agnes Image 2.1 Flash supports image-to-image via the same /images/generations
// endpoint, with input images in an "image" array. We expose this through
// the OpenAI-style /v1/images/edits endpoint.
func (p *Provider) SupportsImageEdit() bool { return true }

// SupportsImageVariation reports image variation capability
func (p *Provider) SupportsImageVariation() bool { return false } // Not yet confirmed

// SupportsVideoGeneration reports video generation capability
func (p *Provider) SupportsVideoGeneration() bool { return true }

// FetchModels returns the list of model names the provider supports. Agnes
// does not expose a /v1/models endpoint, so this returns an error and
// callers are expected to fall back to SupportedModels().
func (p *Provider) FetchModels(ctx context.Context) ([]string, error) {
	return nil, fmt.Errorf("agnes: live model list not supported, use SupportedModels()")
}

// resolveModel maps external model name to Agnes model name
func (p *Provider) resolveModel(externalModel string) string {
	if mapped, ok := p.modelMappings[externalModel]; ok {
		return mapped
	}
	return externalModel
}

// doRequest performs an HTTP request to the Agnes AI API
func (p *Provider) doRequest(ctx context.Context, method, path string, body interface{}) (*http.Response, error) {
	var reqBody io.Reader
	if body != nil {
		jsonData, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request body: %w", err)
		}
		reqBody = bytes.NewReader(jsonData)
		p.logger.Info("Agnes API request body",
			zap.String("path", path),
			zap.String("body", string(jsonData)),
		)
	}

	req, err := http.NewRequestWithContext(ctx, method, p.baseURL+path, reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// ★ 从 context 获取客户端传来的 API Key，直接透传给 Agnes
	apiKey := utils.APIKeyFromCtx(ctx)
	if apiKey == "" {
		apiKey = p.apiKey // fallback: 允许配置默认 key（可选）
	}
	if apiKey == "" {
		return nil, provider.ErrNoAPIKey
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	p.logger.Info("Agnes API request",
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
		p.logger.Error("Agnes API error response",
			zap.Int("status", resp.StatusCode),
			zap.String("body", string(bodyBytes)),
		)
		return nil, &provider.ProviderError{
			StatusCode: resp.StatusCode,
			Body:       string(bodyBytes),
			Provider:   "Agnes",
		}
	}

	// Streaming (SSE) responses — don't buffer the body, just log metadata.
	respCT := resp.Header.Get("Content-Type")
	if strings.Contains(respCT, "text/event-stream") {
		p.logger.Info("Agnes API response (stream)",
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
	p.logger.Info("Agnes API response",
		zap.Int("status", resp.StatusCode),
		zap.String("body", string(bodyBytes)),
	)
	resp.Body = io.NopCloser(bytes.NewReader(bodyBytes))
	return resp, nil
}
