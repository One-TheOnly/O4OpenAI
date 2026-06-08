package moark

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"mime/multipart"
	"strings"

	"github.com/o4openai/internal/model"
	"github.com/o4openai/pkg/utils"
	"go.uber.org/zap"
)

// ============================================================
// Image adapter - text-to-image and image edit.
//
//   - /v1/images/generations: JSON request/response, OpenAI-compatible.
//   - /v1/images/edits: multipart/form-data. The image field can be a
//     URL string, a plain base64 string, or a Data URI. We normalise
//     all three into either a URL form field (if the source is already
//     a URL the gateway can reach) or a file form field (if it's a
//     base64 payload we need to decode and send as binary).
// ============================================================

// ImageGeneration - text-to-image.
func (p *Provider) ImageGeneration(ctx context.Context, req *model.ImageGenerationRequest) (*model.ImageResponse, error) {
	body := p.buildImageGenRequest(req)

	resp, err := p.doJSON(ctx, "POST", "/images/generations", body)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var out model.ImageResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("failed to decode Moark image response: %w", err)
	}
	return &out, nil
}

// ImageEdit - image-to-image (multipart).
func (p *Provider) ImageEdit(ctx context.Context, req *model.ImageEditRequest) (*model.ImageResponse, error) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)

	// Text fields
	if err := w.WriteField("model", p.resolveModel(req.Model)); err != nil {
		return nil, fmt.Errorf("failed to write model field: %w", err)
	}
	if req.Prompt != "" {
		if err := w.WriteField("prompt", req.Prompt); err != nil {
			return nil, fmt.Errorf("failed to write prompt field: %w", err)
		}
	}
	if req.Size != "" {
		if err := w.WriteField("size", req.Size); err != nil {
			return nil, fmt.Errorf("failed to write size field: %w", err)
		}
	}
	if req.ResponseFormat != "" {
		if err := w.WriteField("response_format", req.ResponseFormat); err != nil {
			return nil, fmt.Errorf("failed to write response_format field: %w", err)
		}
	}
	if req.N != nil {
		if err := w.WriteField("n", fmt.Sprintf("%d", *req.N)); err != nil {
			return nil, fmt.Errorf("failed to write n field: %w", err)
		}
	}
	if req.User != "" {
		if err := w.WriteField("user", req.User); err != nil {
			return nil, fmt.Errorf("failed to write user field: %w", err)
		}
	}

	// Image field (single + multi-image collected into one stream).
	imageSources := req.Images
	if len(imageSources) == 0 && req.Image != "" {
		imageSources = []string{req.Image}
	}
	if len(imageSources) == 0 {
		return nil, fmt.Errorf("at least one image is required for image edit")
	}
	// Moark's edit endpoint accepts a single image. We send the first;
	// additional ones are dropped with a warning (similar to the way
	// the OpenAI spec defines it for dall-e-2).
	if err := p.writeImageField(w, "image", imageSources[0]); err != nil {
		return nil, err
	}
	if len(imageSources) > 1 {
		p.logger.Warn("Moark image edit accepts only a single image; dropping extras",
			zap.Int("dropped", len(imageSources)-1),
		)
	}

	// Mask (optional).
	if req.Mask != "" {
		if err := p.writeImageField(w, "mask", req.Mask); err != nil {
			return nil, err
		}
	}

	if err := w.Close(); err != nil {
		return nil, fmt.Errorf("failed to finalize multipart body: %w", err)
	}

	p.logger.Info("Moark image edit (i2i) request",
		zap.String("model", req.Model),
		zap.Int("image_count", len(imageSources)),
	)

	resp, err := p.doRaw(ctx, "POST", "/images/edits", w.FormDataContentType(), &buf)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var out model.ImageResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("failed to decode Moark image edit response: %w", err)
	}
	return &out, nil
}

// ImageVariation is not supported by Moark's public API.
func (p *Provider) ImageVariation(ctx context.Context, req *model.ImageVariationRequest) (*model.ImageResponse, error) {
	return nil, fmt.Errorf("moark: image variation is not supported by the Moark API")
}

// buildImageGenRequest builds the JSON body for /v1/images/generations.
// We use a map so response_format, n, and any extras pass through cleanly.
func (p *Provider) buildImageGenRequest(req *model.ImageGenerationRequest) map[string]interface{} {
	body := map[string]interface{}{
		"model":  p.resolveModel(req.Model),
		"prompt": req.Prompt,
	}
	if req.N != nil {
		body["n"] = *req.N
	}
	if req.Size != "" {
		body["size"] = req.Size
	}
	if req.ResponseFormat != "" {
		body["response_format"] = req.ResponseFormat
	}
	if req.User != "" {
		body["user"] = req.User
	}
	return body
}

// writeImageField writes an "image" (or "mask") part of a multipart form.
// Behaviour:
//   - Empty input: skipped.
//   - HTTP(S) URL: written as a plain string field.
//   - Data URI: base64 payload decoded and written as a file field.
//   - Plain base64 (looks like base64): wrapped and decoded, then written.
func (p *Provider) writeImageField(w *multipart.Writer, field, src string) error {
	src = strings.TrimSpace(src)
	if src == "" {
		return nil
	}
	// Remote URL: pass through as a string.
	if strings.HasPrefix(src, "http://") || strings.HasPrefix(src, "https://") {
		return w.WriteField(field, src)
	}
	// Data URI: decode and send as a file.
	if utils.IsDataURL(src) {
		return writeBase64AsFile(w, field, src)
	}
	// Bare base64: wrap as a data URI first, then send as a file.
	if looksLikeBase64(src) {
		return writeBase64AsFile(w, field, "data:image/png;base64,"+src)
	}
	// Last resort: send as a string (Moark may accept the literal).
	return w.WriteField(field, src)
}

// writeBase64AsFile decodes a Data URI and writes the raw bytes to a
// multipart file part.
func writeBase64AsFile(w *multipart.Writer, field, dataURI string) error {
	const prefix = "base64,"
	idx := strings.Index(dataURI, prefix)
	if idx < 0 {
		return fmt.Errorf("invalid data URI for %s", field)
	}
	raw := dataURI[idx+len(prefix):]
	bytes, err := base64.StdEncoding.DecodeString(raw)
	if err != nil {
		return fmt.Errorf("failed to decode base64 for %s: %w", field, err)
	}
	fw, err := w.CreateFormFile(field, field+".png")
	if err != nil {
		return fmt.Errorf("failed to create form file for %s: %w", field, err)
	}
	if _, err := fw.Write(bytes); err != nil {
		return fmt.Errorf("failed to write %s payload: %w", field, err)
	}
	return nil
}

// looksLikeBase64 returns true for strings that look like a base64
// payload (no "data:" prefix, no scheme, no whitespace, etc.).
func looksLikeBase64(s string) bool {
	if len(s) < 32 {
		return false
	}
	for _, c := range s {
		switch {
		case c >= 'A' && c <= 'Z',
			c >= 'a' && c <= 'z',
			c >= '0' && c <= '9',
			c == '+', c == '/', c == '=':
			// ok
		default:
			return false
		}
	}
	return true
}
