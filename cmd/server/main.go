package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/o4openai/internal/config"
	"github.com/o4openai/internal/handler"
	"github.com/o4openai/internal/model"
	"github.com/o4openai/internal/provider"
	"github.com/o4openai/internal/provider/agnes"
	"github.com/o4openai/internal/provider/moark"
	"github.com/o4openai/pkg/utils"
	"go.uber.org/zap"
)

// newLogger returns a production-grade zap logger.
// Key safety properties:
//   - No stacktraces (no internal file paths leaked)
//   - No caller info (no /Users/... or /opt/homebrew/... paths)
//   - Compact JSON output
func newLogger() *zap.Logger {
	cfg := zap.NewProductionConfig()
	cfg.EncoderConfig.TimeKey = "ts"
	cfg.EncoderConfig.MessageKey = "msg"
	cfg.EncoderConfig.LevelKey = "level"
	cfg.EncoderConfig.CallerKey = "" // suppress caller (would expose local paths)
	cfg.EncoderConfig.StacktraceKey = ""
	cfg.DisableStacktrace = true // never include stacktraces
	cfg.DisableCaller = true
	cfg.Sampling = nil // log every error
	logger, _ := cfg.Build()
	return logger
}

func main() {
	configPath := flag.String("config", "", "Path to config file")
	port := flag.Int("port", 0, "Override server port")
	flag.Parse()

	logger := newLogger()
	defer logger.Sync()

	cfg, err := config.Load(*configPath, logger)
	if err != nil {
		logger.Fatal("Failed to load config", zap.Error(err))
	}

	if *port != 0 {
		cfg.Server.Port = *port
	}

	gin.SetMode(cfg.Server.Mode)

	// Setup Base64 handler for converting base64 images to temp URLs
	publicURL := cfg.Gateway.PublicURL
	ttl := time.Duration(cfg.Gateway.TempImageTTL) * time.Minute
	cleanupInterval := time.Duration(cfg.Gateway.TempImageCleanupInterval) * time.Minute
	var base64Handler *utils.Base64Handler
	if publicURL != "" {
		base64Handler = utils.NewBase64Handler(publicURL, ttl)
		base64Handler.StartCleanupRoutine(cleanupInterval)
		logger.Info("Base64 handler enabled",
			zap.String("public_url", publicURL),
			zap.Duration("ttl", ttl),
		)
	} else {
		logger.Warn("gateway.public_url not set! Base64 image conversion disabled. " +
			"Set it to your public URL (e.g., https://your-domain.com) to enable " +
			"image/video generation with base64 inputs.")
	}

	registry := provider.NewRegistry()

	// fetchTimeout caps how long we'll wait for an upstream /v1/models
	// round-trip at startup. 15s is plenty for any reasonable provider.
	const fetchTimeout = 15 * time.Second

	// resolveModelConfigs builds the model mapping list for a provider.
	//   1. If the user pinned models in config.yaml, use that.
	//   2. Otherwise, ask the provider to FetchModels() from upstream.
	//   3. If that fails, fall back to the provider's hardcoded
	//      SupportedModels() list. This keeps the server usable even
	//      when the upstream is down at boot.
	resolveModelConfigs := func(p model.Provider, pc config.ProviderConfig) []model.ModelMapping {
		if len(pc.Models) > 0 {
			out := make([]model.ModelMapping, 0, len(pc.Models))
			for _, mc := range pc.Models {
				out = append(out, model.ModelMapping{
					ExternalModel: mc.ExternalModel,
					ProviderModel: mc.ProviderModel,
					Capabilities:  mc.Capabilities,
				})
			}
			return out
		}

		// No explicit model list: try to fetch a live list from the provider.
		logger.Info("No models pinned in config; fetching live model list",
			zap.String("provider", p.Name()),
		)
		fctx, cancel := context.WithTimeout(context.Background(), fetchTimeout)
		// If the provider was constructed with a fallback key, surface it
		// in the context so FetchModels can authenticate.
		fctx = utils.WithAPIKey(fctx, "")
		ids, err := p.FetchModels(fctx)
		cancel()
		if err != nil || len(ids) == 0 {
			if err != nil {
				logger.Warn("FetchModels failed; falling back to SupportedModels()",
					zap.String("provider", p.Name()),
					zap.Error(err),
				)
			}
			fallback := p.SupportedModels()
			ids = make([]string, 0, len(fallback))
			for _, m := range fallback {
				ids = append(ids, m.ID)
			}
		}

		out := make([]model.ModelMapping, 0, len(ids))
		for _, id := range ids {
			out = append(out, model.ModelMapping{
				ExternalModel: id,
				ProviderModel: id,
				// Capabilities left empty; the provider's Supports* methods
				// are still consulted by the handler when picking a route.
			})
		}
		logger.Info("Auto-discovered models from provider",
			zap.String("provider", p.Name()),
			zap.Int("count", len(out)),
		)
		return out
	}

	// Register providers
	// ★ 不需要配置 API Key！客户端传什么 key，网关就透传给 Provider
	for name, pc := range cfg.Providers {
		if !pc.Enabled {
			logger.Info("Skipping disabled provider", zap.String("provider", name))
			continue
		}

		switch name {
		case "agnes":
			p := agnes.NewProvider(pc.APIKey, pc.BaseURL, logger) // api_key 可选（作为 fallback）
			mappings := make(map[string]string)
			for _, mc := range pc.Models {
				mappings[mc.ExternalModel] = mc.ProviderModel
			}
			p.SetModelMappings(mappings)
			p.SetBase64Handler(base64Handler)
			modelConfigs := resolveModelConfigs(p, pc)
			if err := registry.Register(p, modelConfigs); err != nil {
				logger.Fatal("Failed to register Agnes provider", zap.Error(err))
			}
			logger.Info("Registered Agnes AI provider",
				zap.String("base_url", pc.BaseURL),
				zap.Int("models", len(modelConfigs)),
			)
		case "moark":
			p := moark.NewProvider(pc.APIKey, pc.BaseURL, logger) // api_key 可选（作为 fallback）
			mappings := make(map[string]string)
			for _, mc := range pc.Models {
				mappings[mc.ExternalModel] = mc.ProviderModel
			}
			p.SetModelMappings(mappings)
			p.SetBase64Handler(base64Handler)
			modelConfigs := resolveModelConfigs(p, pc)
			if err := registry.Register(p, modelConfigs); err != nil {
				logger.Fatal("Failed to register Moark provider", zap.Error(err))
			}
			logger.Info("Registered Moark provider",
				zap.String("base_url", pc.BaseURL),
				zap.Int("models", len(modelConfigs)),
			)
		default:
			logger.Warn("Unknown provider type, skipping", zap.String("provider", name))
		}
	}

	engine := gin.New()
	engine.Use(gin.Recovery())

	// ★ 不需要认证中间件！客户端的 API Key 在 handler 层透传给 Provider
	handler.SetupRouter(engine, registry, base64Handler, logger)

	printBanner(cfg, registry, logger)

	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	srv := &http.Server{Addr: addr, Handler: engine}

	go func() {
		logger.Info("Server starting", zap.String("addr", addr))
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal("Server failed", zap.Error(err))
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	logger.Info("Shutting down server...")
}

func printBanner(cfg *config.Config, registry *provider.Registry, logger *zap.Logger) {
	fmt.Println()
	fmt.Println("  ============================================")
	fmt.Println("       O4OpenAI API Gateway")
	fmt.Println("       OpenAI-Compatible API Proxy Platform")
	fmt.Println("  ============================================")
	fmt.Printf("  Server:     http://%s:%d\n", cfg.Server.Host, cfg.Server.Port)
	fmt.Printf("  Public URL: %s\n", cfg.Gateway.PublicURL)

	totalModels := 0
	for _, name := range registry.GetAllProviders() {
		models := registry.ListModelsByProvider(name)
		totalModels += len(models)
		fmt.Printf("\n  [%s] %d models:\n", name, len(models))
		for _, m := range models {
			fmt.Printf("    - %s\n", m.ID)
		}
	}
	fmt.Printf("\n  Total: %d models, %d providers\n", totalModels, len(registry.GetAllProviders()))

	fmt.Println()
	fmt.Println("  Routing:")
	fmt.Println("    /v1/chat/completions           -> auto (by model name)")
	fmt.Println("    /v1/images/generations         -> auto")
	fmt.Println("    /v1/videos/generations         -> auto")
	fmt.Println("    /v1/messages                   -> Anthropic Messages API")
	for _, name := range registry.GetAllProviders() {
		fmt.Printf("    /%s/v1/chat/completions     -> force %s\n", name, name)
		fmt.Printf("    /%s/v1/images/generations   -> force %s\n", name, name)
		fmt.Printf("    /%s/v1/videos/generations   -> force %s\n", name, name)
		fmt.Printf("    /%s/v1/messages             -> force %s (Anthropic)\n", name, name)
	}
	fmt.Println()
	fmt.Println("  Clients just change the base URL to switch providers:")
	fmt.Printf("    base_url = \"http://localhost:%d\"       -> auto\n", cfg.Server.Port)
	for _, name := range registry.GetAllProviders() {
		fmt.Printf("    base_url = \"http://localhost:%d/%s\"    -> force %s\n", cfg.Server.Port, name, name)
	}
	fmt.Println()
}
