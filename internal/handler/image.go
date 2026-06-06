package handler

import (
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/o4openai/internal/model"
	"github.com/o4openai/internal/provider"
	"github.com/o4openai/pkg/utils"
	"go.uber.org/zap"
)

// Suppress unused import

// ImageHandler handles image-related requests
type ImageHandler struct {
	registry       *provider.Registry
	logger         *zap.Logger
	base64Handler  *utils.Base64Handler
	forcedProvider string
}

// NewImageHandler creates a new image handler
func NewImageHandler(registry *provider.Registry, base64Handler *utils.Base64Handler, logger *zap.Logger, forcedProvider string) *ImageHandler {
	return &ImageHandler{
		registry:       registry,
		logger:         logger,
		base64Handler:  base64Handler,
		forcedProvider: forcedProvider,
	}
}

// resolveProvider finds the provider by forced name or model name
func (h *ImageHandler) resolveProvider(modelName string) (model.Provider, error) {
	if h.forcedProvider != "" {
		return h.registry.GetProvider(h.forcedProvider)
	}
	if modelName != "" {
		return h.registry.GetProviderForModel(modelName)
	}
	return h.registry.GetProviderForModel("dall-e-2") // fallback default
}

// HandleGenerate handles POST /v1/images/generations
func (h *ImageHandler) HandleGenerate(c *gin.Context) {
	contentType := c.GetHeader("Content-Type")

	var req model.ImageGenerationRequest
	var err error

	if strings.HasPrefix(contentType, "multipart/form-data") {
		err = h.parseMultipartGenerate(c, &req)
	} else {
		err = c.ShouldBindJSON(&req)
	}

	if err != nil {
		c.JSON(http.StatusBadRequest, model.ErrorResponse{
			Error: model.ErrorDetail{
				Message: fmt.Sprintf("Invalid request: %v", err),
				Type:    "invalid_request_error",
				Code:    "invalid_json",
			},
		})
		return
	}

	h.logger.Info("Image generation request",
		zap.String("model", req.Model),
		zap.String("size", req.Size),
		zap.String("response_format", req.ResponseFormat),
	)

	// Determine which model to use
	modelName := req.Model
	if modelName == "" {
		modelName = "dall-e-2" // OpenAI default
	}

	p, err := h.resolveProvider(modelName)
	if err != nil {
		c.JSON(http.StatusBadRequest, model.ErrorResponse{
			Error: model.ErrorDetail{
				Message: fmt.Sprintf("Model %q not found for image generation", modelName),
				Type:    "invalid_request_error",
				Code:    "model_not_found",
			},
		})
		return
	}

	if !p.SupportsImageGeneration() {
		c.JSON(http.StatusBadRequest, model.ErrorResponse{
			Error: model.ErrorDetail{
				Message: fmt.Sprintf("Provider %q does not support image generation", p.Name()),
				Type:    "invalid_request_error",
				Code:    "unsupported_capability",
			},
		})
		return
	}

	resp, err := p.ImageGeneration(ctxWithKey(c), &req)
	if err != nil {
		h.logger.Error("Image generation failed", zap.Error(err))
		respondProviderError(c, "Image generation", err)
		return
	}

	c.JSON(http.StatusOK, resp)
}

// HandleEdit handles POST /v1/images/edits
func (h *ImageHandler) HandleEdit(c *gin.Context) {
	contentType := c.GetHeader("Content-Type")

	h.logger.Info("HandleEdit received request",
		zap.String("content_type", contentType),
	)

	var req model.ImageEditRequest
	var err error

	if strings.HasPrefix(contentType, "multipart/form-data") {
		h.logger.Info("HandleEdit: using multipart parser")
		err = h.parseMultipartEdit(c, &req)
	} else {
		h.logger.Info("HandleEdit: using JSON parser")
		err = c.ShouldBindJSON(&req)
	}

	h.logger.Info("HandleEdit: parsed request",
		zap.String("model", req.Model),
		zap.Bool("has_image", req.Image != ""),
		zap.Int("images_count", len(req.Images)),
		zap.String("image_preview", func() string {
			if len(req.Image) > 100 {
				return req.Image[:100] + "..."
			}
			return req.Image
		}()),
	)

	if err != nil {
		c.JSON(http.StatusBadRequest, model.ErrorResponse{
			Error: model.ErrorDetail{
				Message: fmt.Sprintf("Invalid request: %v", err),
				Type:    "invalid_request_error",
				Code:    "invalid_json",
			},
		})
		return
	}

	modelName := req.Model
	if modelName == "" {
		modelName = "dall-e-2"
	}

	p, err := h.resolveProvider(modelName)
	if err != nil {
		c.JSON(http.StatusBadRequest, model.ErrorResponse{
			Error: model.ErrorDetail{
				Message: fmt.Sprintf("Model %q not found for image editing", modelName),
				Type:    "invalid_request_error",
				Code:    "model_not_found",
			},
		})
		return
	}

	if !p.SupportsImageEdit() {
		c.JSON(http.StatusBadRequest, model.ErrorResponse{
			Error: model.ErrorDetail{
				Message: fmt.Sprintf("Provider %q does not support image editing", p.Name()),
				Type:    "invalid_request_error",
				Code:    "unsupported_capability",
			},
		})
		return
	}

	resp, err := p.ImageEdit(ctxWithKey(c), &req)
	if err != nil {
		h.logger.Error("Image edit failed", zap.Error(err))
		respondProviderError(c, "Image edit", err)
		return
	}

	c.JSON(http.StatusOK, resp)
}

// HandleVariation handles POST /v1/images/variations
func (h *ImageHandler) HandleVariation(c *gin.Context) {
	contentType := c.GetHeader("Content-Type")

	var req model.ImageVariationRequest
	var err error

	if strings.HasPrefix(contentType, "multipart/form-data") {
		err = h.parseMultipartVariation(c, &req)
	} else {
		err = c.ShouldBindJSON(&req)
	}

	if err != nil {
		c.JSON(http.StatusBadRequest, model.ErrorResponse{
			Error: model.ErrorDetail{
				Message: fmt.Sprintf("Invalid request: %v", err),
				Type:    "invalid_request_error",
				Code:    "invalid_json",
			},
		})
		return
	}

	modelName := req.Model
	if modelName == "" {
		modelName = "dall-e-2"
	}

	p, err := h.resolveProvider(modelName)
	if err != nil {
		c.JSON(http.StatusBadRequest, model.ErrorResponse{
			Error: model.ErrorDetail{
				Message: fmt.Sprintf("Model %q not found for image variations", modelName),
				Type:    "invalid_request_error",
				Code:    "model_not_found",
			},
		})
		return
	}

	if !p.SupportsImageVariation() {
		c.JSON(http.StatusBadRequest, model.ErrorResponse{
			Error: model.ErrorDetail{
				Message: fmt.Sprintf("Provider %q does not support image variations", p.Name()),
				Type:    "invalid_request_error",
				Code:    "unsupported_capability",
			},
		})
		return
	}

	resp, err := p.ImageVariation(c.Request.Context(), &req)
	if err != nil {
		h.logger.Error("Image variation failed", zap.Error(err))
		respondProviderError(c, "Image variation", err)
		return
	}

	c.JSON(http.StatusOK, resp)
}

// parseMultipartGenerate parses multipart form data for image generation
func (h *ImageHandler) parseMultipartGenerate(c *gin.Context, req *model.ImageGenerationRequest) error {
	form, err := c.MultipartForm()
	if err != nil {
		return err
	}

	// Parse simple fields
	if values := form.Value["model"]; len(values) > 0 {
		req.Model = values[0]
	}
	if values := form.Value["prompt"]; len(values) > 0 {
		req.Prompt = values[0]
	}
	if values := form.Value["size"]; len(values) > 0 {
		req.Size = values[0]
	}
	if values := form.Value["quality"]; len(values) > 0 {
		req.Quality = values[0]
	}
	if values := form.Value["response_format"]; len(values) > 0 {
		req.ResponseFormat = values[0]
	}
	if values := form.Value["style"]; len(values) > 0 {
		req.Style = values[0]
	}
	if values := form.Value["user"]; len(values) > 0 {
		req.User = values[0]
	}
	if values := form.Value["n"]; len(values) > 0 {
		var n int
		fmt.Sscanf(values[0], "%d", &n)
		req.N = &n
	}

	return nil
}

// parseMultipartEdit parses multipart form data for image editing.
// Supports both single and multiple image file uploads.
// When multiple files are provided under the "image" key, all are collected
// into req.Images for downstream multi-image processing (e.g. ArcReel SDK).
func (h *ImageHandler) parseMultipartEdit(c *gin.Context, req *model.ImageEditRequest) error {
	contentType := c.GetHeader("Content-Type")
	h.logger.Info("parseMultipartEdit called",
		zap.String("content_type", contentType),
	)

	form, err := c.MultipartForm()
	if err != nil {
		return err
	}

	// Debug: log all form fields
	for key, vals := range form.Value {
		for i, v := range vals {
			preview := v
			if len(preview) > 100 {
				preview = preview[:100] + "..."
			}
			h.logger.Info("form value field",
				zap.String("key", key),
				zap.Int("index", i),
				zap.Int("total_len", len(v)),
				zap.String("preview", preview),
			)
		}
	}
	for key, files := range form.File {
		for i, f := range files {
			h.logger.Info("form file field",
				zap.String("key", key),
				zap.Int("index", i),
				zap.String("filename", f.Filename),
				zap.Int64("size", f.Size),
			)
		}
	}

	if values := form.Value["model"]; len(values) > 0 {
		req.Model = values[0]
	}
	if values := form.Value["prompt"]; len(values) > 0 {
		req.Prompt = values[0]
	}
	if values := form.Value["size"]; len(values) > 0 {
		req.Size = values[0]
	}
	if values := form.Value["response_format"]; len(values) > 0 {
		req.ResponseFormat = values[0]
	}
	if values := form.Value["n"]; len(values) > 0 {
		var n int
		fmt.Sscanf(values[0], "%d", &n)
		req.N = &n
	}
	if values := form.Value["quality"]; len(values) > 0 {
		req.Quality = values[0]
	}

	// Parse image inputs — support multiple field names and both file/value types.
	// OpenAI Python SDK with multiple images uses "image[]"; other SDKs use "image".
	if err := h.collectImageInputs(form, req); err != nil {
		return err
	}

	if files := form.File["mask"]; len(files) > 0 {
		file, err := files[0].Open()
		if err != nil {
			return fmt.Errorf("failed to open mask file: %w", err)
		}
		defer file.Close()
		data, err := io.ReadAll(file)
		if err != nil {
			return fmt.Errorf("failed to read mask file: %w", err)
		}
		req.Mask = encodeToBase64String(data)
	}

	return nil
}

// parseMultipartVariation parses multipart form data for image variations
func (h *ImageHandler) parseMultipartVariation(c *gin.Context, req *model.ImageVariationRequest) error {
	form, err := c.MultipartForm()
	if err != nil {
		return err
	}

	if values := form.Value["model"]; len(values) > 0 {
		req.Model = values[0]
	}
	if values := form.Value["size"]; len(values) > 0 {
		req.Size = values[0]
	}
	if values := form.Value["response_format"]; len(values) > 0 {
		req.ResponseFormat = values[0]
	}
	if values := form.Value["n"]; len(values) > 0 {
		var n int
		fmt.Sscanf(values[0], "%d", &n)
		req.N = &n
	}

	if files := form.File["image"]; len(files) > 0 {
		file, err := files[0].Open()
		if err != nil {
			return fmt.Errorf("failed to open image file: %w", err)
		}
		defer file.Close()
		data, err := io.ReadAll(file)
		if err != nil {
			return fmt.Errorf("failed to read image file: %w", err)
		}
		req.Image = encodeToBase64String(data)
	}

	return nil
}

// collectImageInputs populates req.Image / req.Images from the multipart form.
// It supports multiple field names ("image", "image[]") and both file uploads
// (form.File) and text fields (form.Value) carrying URLs or base64 strings.
// OpenAI Python SDK uses "image[]" for multi-image uploads; some other SDKs
// (e.g. ArcReel) use plain "image".
func (h *ImageHandler) collectImageInputs(form *multipart.Form, req *model.ImageEditRequest) error {
	// Try file uploads first (the OpenAI standard).
	// Check both "image" and "image[]" field names.
	var files []*multipart.FileHeader
	for _, key := range []string{"image", "image[]"} {
		if f := form.File[key]; len(f) > 0 {
			files = f
			break
		}
	}

	if len(files) > 0 {
		if len(files) > 16 {
			return fmt.Errorf("too many images: maximum 16 allowed, got %d", len(files))
		}
		if len(files) == 1 {
			// Single file: backward-compatible path
			file, err := files[0].Open()
			if err != nil {
				return fmt.Errorf("failed to open image file: %w", err)
			}
			defer file.Close()
			data, err := io.ReadAll(file)
			if err != nil {
				return fmt.Errorf("failed to read image file: %w", err)
			}
			req.Image = encodeToBase64String(data)
			return nil
		}
		// Multiple files: populate Images slice
		images := make([]string, 0, len(files))
		for i, fh := range files {
			file, err := fh.Open()
			if err != nil {
				return fmt.Errorf("failed to open image file %d: %w", i, err)
			}
			data, err := io.ReadAll(file)
			file.Close()
			if err != nil {
				return fmt.Errorf("failed to read image file %d: %w", i, err)
			}
			images = append(images, encodeToBase64String(data))
		}
		req.Images = images
		// Also set Image to the first for any code that only checks Image
		req.Image = images[0]
		return nil
	}

	// Fall back to text fields (URL or base64 string), again checking both names.
	var values []string
	for _, key := range []string{"image", "image[]"} {
		if v := form.Value[key]; len(v) > 0 {
			values = v
			break
		}
	}
	if len(values) > 0 {
		if len(values) == 1 {
			req.Image = values[0]
		} else {
			req.Images = values
			req.Image = values[0]
		}
	}

	return nil
}
