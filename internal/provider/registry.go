package provider

import (
	"fmt"
	"sync"
	"time"

	"github.com/o4openai/internal/model"
)

// ============================================================
// Provider Registry - manages all registered providers
// Routes requests to the correct provider based on model name
// ============================================================

// Registry manages all provider instances and routes by model name
type Registry struct {
	mu         sync.RWMutex
	providers  map[string]model.Provider    // keyed by provider name
	modelMap   map[string]model.Provider    // keyed by external model name
	modelNames map[string]string            // external model name -> provider name
	modelInfos map[string][]model.ModelInfo // provider name -> model list
}

// NewRegistry creates a new provider registry
func NewRegistry() *Registry {
	return &Registry{
		providers:  make(map[string]model.Provider),
		modelMap:   make(map[string]model.Provider),
		modelNames: make(map[string]string),
		modelInfos: make(map[string][]model.ModelInfo),
	}
}

// Register adds a provider to the registry
func (r *Registry) Register(p model.Provider, configs []model.ModelMapping) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	name := p.Name()
	if _, exists := r.providers[name]; exists {
		return fmt.Errorf("provider %q already registered", name)
	}

	r.providers[name] = p

	// Build the model info list by merging SupportedModels (rich metadata)
	// with the resolved configs (may include models discovered via FetchModels
	// or pinned in config.yaml that aren't in the hardcoded list).
	baseInfos := p.SupportedModels()
	infoMap := make(map[string]model.ModelInfo, len(baseInfos))
	for _, m := range baseInfos {
		infoMap[m.ID] = m
	}
	// Add any models from configs that aren't already in the base list.
	for _, cfg := range configs {
		if _, exists := infoMap[cfg.ExternalModel]; !exists {
			infoMap[cfg.ExternalModel] = model.ModelInfo{
				ID:      cfg.ExternalModel,
				Object:  "model",
				Created: time.Now().Unix(),
				OwnedBy: name,
			}
		}
	}
	infos := make([]model.ModelInfo, 0, len(infoMap))
	for _, m := range infoMap {
		infos = append(infos, m)
	}
	r.modelInfos[name] = infos

	// Build model name mappings (used for routing requests to the right provider)
	for _, cfg := range configs {
		r.modelMap[cfg.ExternalModel] = p
		r.modelNames[cfg.ExternalModel] = name
	}

	return nil
}

// GetProviderForModel returns the provider that handles the given model name
func (r *Registry) GetProviderForModel(modelName string) (model.Provider, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	p, ok := r.modelMap[modelName]
	if !ok {
		return nil, fmt.Errorf("no provider found for model %q", modelName)
	}
	return p, nil
}

// GetProvider returns a provider by name
func (r *Registry) GetProvider(name string) (model.Provider, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	p, ok := r.providers[name]
	if !ok {
		return nil, fmt.Errorf("provider %q not found", name)
	}
	return p, nil
}

// ListModels returns all available models across all providers
func (r *Registry) ListModels() []model.ModelInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var all []model.ModelInfo
	for _, infos := range r.modelInfos {
		all = append(all, infos...)
	}
	return all
}

// ListModelsByProvider returns models belonging to a specific provider.
func (r *Registry) ListModelsByProvider(providerName string) []model.ModelInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()

	return r.modelInfos[providerName]
}

// GetAllProviders returns all registered provider names
func (r *Registry) GetAllProviders() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.providers))
	for name := range r.providers {
		names = append(names, name)
	}
	return names
}
