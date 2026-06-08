package handler

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/o4openai/internal/model"
	"github.com/o4openai/internal/provider"
	"go.uber.org/zap"
)

// ============================================================
// Models handler - lists available models
// ============================================================

// ModelsHandler handles model listing requests
type ModelsHandler struct {
	registry       *provider.Registry
	logger         *zap.Logger
	forcedProvider string // when non-empty, only list models for this provider
}

// NewModelsHandler creates a new models handler
func NewModelsHandler(registry *provider.Registry, logger *zap.Logger, forcedProvider string) *ModelsHandler {
	return &ModelsHandler{
		registry:       registry,
		logger:         logger,
		forcedProvider: forcedProvider,
	}
}

// HandleList handles GET /v1/models
func (h *ModelsHandler) HandleList(c *gin.Context) {
	var models []model.ModelInfo
	if h.forcedProvider != "" {
		models = h.registry.ListModelsByProvider(h.forcedProvider)
	} else {
		models = h.registry.ListModels()
	}

	// If no models registered, return default list
	if len(models) == 0 {
		models = []model.ModelInfo{
			{
				ID:      "agnes-1.5-flash",
				Object:  "model",
				Created: time.Now().Unix(),
				OwnedBy: "agnes",
			},
		}
	}

	c.JSON(http.StatusOK, model.ModelListResponse{
		Object: "list",
		Data:   models,
	})
}

// HandleRetrieve handles GET /v1/models/:model
func (h *ModelsHandler) HandleRetrieve(c *gin.Context) {
	modelID := c.Param("model")

	// Check if this model exists
	models := h.registry.ListModels()
	for _, m := range models {
		if m.ID == modelID {
			c.JSON(http.StatusOK, m)
			return
		}
	}

	c.JSON(http.StatusNotFound, model.ErrorResponse{
		Error: model.ErrorDetail{
			Message: "Model not found: " + modelID,
			Type:    "invalid_request_error",
			Code:    "model_not_found",
		},
	})
}
