package model

import (
	"context"
	"io"
)

// ============================================================
// Provider Interface - the core abstraction for all providers
// Each provider implements this interface to handle conversion
// between OpenAI format and its native format.
// ============================================================

// Provider is the interface that all third-party providers must implement.
// This is the key abstraction that makes the system extensible.
type Provider interface {
	// Name returns the provider's unique identifier
	Name() string

	// SupportedModels returns the list of models this provider supports
	SupportedModels() []ModelInfo

	// Chat
	ChatCompletion(ctx context.Context, req *ChatCompletionRequest) (*ChatCompletionResponse, error)
	ChatCompletionStream(ctx context.Context, req *ChatCompletionRequest) (io.ReadCloser, error)

	// Images
	ImageGeneration(ctx context.Context, req *ImageGenerationRequest) (*ImageResponse, error)
	ImageEdit(ctx context.Context, req *ImageEditRequest) (*ImageResponse, error)
	ImageVariation(ctx context.Context, req *ImageVariationRequest) (*ImageResponse, error)

	// Video
	VideoGeneration(ctx context.Context, req *VideoGenerationRequest) (*VideoResponse, error)
	VideoRetrieve(ctx context.Context, videoID string) (*VideoResponse, error)

	// Capabilities - report what this provider supports
	SupportsChat() bool
	SupportsImageGeneration() bool
	SupportsImageEdit() bool
	SupportsImageVariation() bool
	SupportsVideoGeneration() bool

	// FetchModels optionally discovers the live list of model names from
	// the upstream API (e.g. by calling GET /v1/models). Implementations
	// that don't expose a model list endpoint should return an error;
	// the registry will then fall back to SupportedModels().
	FetchModels(ctx context.Context) ([]string, error)
}

// ProviderConfig holds the configuration for a provider
type ProviderConfig struct {
	Name     string            `yaml:"name" json:"name"`
	Type     string            `yaml:"type" json:"type"` // "agnes", "openai", "custom", etc.
	BaseURL  string            `yaml:"base_url" json:"base_url"`
	APIKey   string            `yaml:"api_key" json:"api_key"`
	Models   []ModelMapping    `yaml:"models" json:"models"`
	Extra    map[string]string `yaml:"extra" json:"extra"`
	Enabled  bool              `yaml:"enabled" json:"enabled"`
	Priority int               `yaml:"priority" json:"priority"` // Higher = preferred
}

// ModelMapping maps an external model name to the provider's internal model name
type ModelMapping struct {
	ExternalModel string   `yaml:"external_model" json:"external_model"` // Name exposed to clients
	ProviderModel string   `yaml:"provider_model" json:"provider_model"` // Name sent to provider
	Capabilities  []string `yaml:"capabilities" json:"capabilities"`     // chat, image_gen, image_edit, image_var, video
}
