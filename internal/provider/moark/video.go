package moark

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/o4openai/internal/model"
	"github.com/o4openai/pkg/utils"
	"go.uber.org/zap"
)

// ============================================================
// Video adapter - async task pattern.
//
// Moark exposes separate endpoints for different video generation modes:
//
//   POST /v1/async/videos/generations      -> text-to-video (model + prompt)
//   POST /v1/async/videos/image-to-video   -> image-to-video (model + prompt + image_url)
//   GET  /v1/task/{task_id}                -> poll status; when status ==
//                                             "succeeded" the output field
//                                             holds the result (typically
//                                             {"url": "<video-url>"}).
//
// The OpenAI gateway video interface is synchronous, so we model the
// flow like this:
//
//   - VideoGeneration submits the job and returns immediately with
//     status="queued" and the task id as the response id.
//   - VideoRetrieve polls the task endpoint and maps the result into
//     an OpenAI-shaped VideoResponse. The handler then loops on
//     /v1/videos/{id} until status is "completed"/"failed".
// ============================================================

// VideoGeneration submits a new video generation task.
// Moark has separate endpoints:
//   - Text-to-video: POST /async/videos/generations       (model + prompt)
//   - Image-to-video: POST /async/videos/image-to-video   (model + prompt + image_url)
func (p *Provider) VideoGeneration(ctx context.Context, req *model.VideoGenerationRequest) (*model.VideoResponse, error) {
	body, hasImage, err := p.buildVideoRequest(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to build video request: %w", err)
	}

	// Choose the correct endpoint based on whether an image input is present.
	endpoint := "/async/videos/generations"
	if hasImage {
		endpoint = "/async/videos/image-to-video"
	}

	resp, err := p.doJSON(ctx, "POST", endpoint, body)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var task MoarkTaskResponse
	if err := json.NewDecoder(resp.Body).Decode(&task); err != nil {
		return nil, fmt.Errorf("failed to decode Moark video submit response: %w", err)
	}

	return convertTaskToVideoResponse(&task, p.resolveModel(req.Model)), nil
}

// VideoRetrieve fetches the latest status of a previously-submitted task.
func (p *Provider) VideoRetrieve(ctx context.Context, videoID string) (*model.VideoResponse, error) {
	if videoID == "" {
		return nil, fmt.Errorf("moark: video id is required")
	}

	resp, err := p.doJSON(ctx, "GET", "/task/"+videoID, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var task MoarkTaskResponse
	if err := json.NewDecoder(resp.Body).Decode(&task); err != nil {
		return nil, fmt.Errorf("failed to decode Moark task response: %w", err)
	}
	return convertTaskToVideoResponse(&task, ""), nil
}

// buildVideoRequest assembles the JSON body for video generation endpoints.
// Returns the body map, whether an image input was provided, and any error.
//
// Moark has two distinct endpoints with different schemas:
//   - Text-to-video (/async/videos/generations): accepts only model + prompt.
//   - Image-to-video (/async/videos/image-to-video): accepts model + prompt + image_url.
//
// The gateway's VideoGenerationRequest is OpenAI-shaped (input = []InputItem),
// so we collapse it into the appropriate Moark schema.
func (p *Provider) buildVideoRequest(ctx context.Context, req *model.VideoGenerationRequest) (map[string]interface{}, bool, error) {
	body := map[string]interface{}{
		"model": p.resolveModel(req.Model),
	}

	// Collect prompt and image URLs from the OpenAI-shaped input list.
	var promptParts []string
	var imageURLs []string
	for _, item := range req.Input {
		switch item.Type {
		case "text":
			if item.Text != "" {
				promptParts = append(promptParts, item.Text)
			}
		case "image":
			url, err := p.resolveImageURL(ctx, item.Image)
			if err != nil {
				return nil, false, err
			}
			if url != "" {
				imageURLs = append(imageURLs, url)
			}
		}
	}

	// Prompt
	switch {
	case len(promptParts) > 0 && req.Instructions != "":
		body["prompt"] = req.Instructions + "\n" + strings.Join(promptParts, "\n")
	case len(promptParts) > 0:
		body["prompt"] = strings.Join(promptParts, "\n")
	case req.Instructions != "":
		body["prompt"] = req.Instructions
	}

	// Image inputs — only for image-to-video endpoint.
	// Moark uses "image_url" as the field name (not "image").
	// Conservative: pass only the first image; warn on extras.
	hasImage := len(imageURLs) > 0
	if hasImage {
		body["image_url"] = imageURLs[0]
		if len(imageURLs) > 1 {
			p.logger.Warn("Moark image-to-video accepts a single image; dropping extras",
				zap.Int("dropped", len(imageURLs)-1),
			)
		}
	}

	return body, hasImage, nil
}

// resolveImageURL converts a Data URI or plain base64 string into a URL
// Moark can fetch. HTTP(S) URLs are returned as-is.
func (p *Provider) resolveImageURL(ctx context.Context, raw string) (string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", nil
	}
	if strings.HasPrefix(raw, "http://") || strings.HasPrefix(raw, "https://") {
		return raw, nil
	}
	if utils.IsDataURL(raw) || looksLikeBase64(raw) {
		if p.base64Handler == nil {
			return "", fmt.Errorf("base64 image provided but Base64Handler is not configured")
		}
		if !utils.IsDataURL(raw) {
			raw = "data:image/png;base64," + raw
		}
		reqCtx := utils.RequestContextFromCtx(ctx)
		url, _, err := p.base64Handler.ConvertDataURL(raw, reqCtx)
		if err != nil {
			return "", fmt.Errorf("failed to convert base64 image to temp URL: %w", err)
		}
		return url, nil
	}
	return raw, nil
}

// convertTaskToVideoResponse maps a Moark task payload into the gateway's
// OpenAI-shaped VideoResponse.
func convertTaskToVideoResponse(task *MoarkTaskResponse, modelFallback string) *model.VideoResponse {
	resp := &model.VideoResponse{
		ID:        task.TaskID,
		Object:    "video",
		CreatedAt: task.CreatedAt,
		Status:    normalizeVideoStatus(task.Status),
		Model:     modelFallback,
	}

	if task.Status == "succeeded" || task.Status == "completed" {
		if url := extractOutputURL(task.Output); url != "" {
			mime := "video/mp4"
			if strings.HasSuffix(strings.ToLower(url), ".webm") {
				mime = "video/webm"
			}
			resp.Output = []model.VideoOutput{{
				Type:     "url",
				URL:      url,
				Duration: 0,
				MimeType: mime,
			}}
		}
	}

	if task.Error != nil {
		resp.Error = &model.VideoError{
			Code:    task.Error.Code,
			Message: task.Error.Message,
		}
	}
	return resp
}

// extractOutputURL pulls a URL string out of the free-form output object.
// Moark typically returns {"url": "..."} for video, but it may also be
// nested under "data" / "videos" depending on the task type.
func extractOutputURL(output map[string]interface{}) string {
	if output == nil {
		return ""
	}
	if v, ok := output["url"].(string); ok && v != "" {
		return v
	}
	if v, ok := output["video_url"].(string); ok && v != "" {
		return v
	}
	if v, ok := output["output_url"].(string); ok && v != "" {
		return v
	}
	// Nested under data: [{"url": "..."}]
	if data, ok := output["data"].([]interface{}); ok {
		for _, item := range data {
			if m, ok := item.(map[string]interface{}); ok {
				if v, ok := m["url"].(string); ok && v != "" {
					return v
				}
			}
		}
	}
	// Nested under videos: [{...}]
	if videos, ok := output["videos"].([]interface{}); ok {
		for _, item := range videos {
			if m, ok := item.(map[string]interface{}); ok {
				if v, ok := m["url"].(string); ok && v != "" {
					return v
				}
			}
		}
	}
	return ""
}

// normalizeVideoStatus maps Moark task statuses to the OpenAI Video API
// status values used by the gateway: queued, in_progress, completed,
// failed, expired.
func normalizeVideoStatus(s string) string {
	switch strings.ToLower(s) {
	case "waiting", "pending", "queued":
		return "queued"
	case "running", "processing", "in_progress", "started", "node_started":
		return "in_progress"
	case "succeeded", "success", "completed", "node_finished", "workflow_finished":
		return "completed"
	case "failed", "failure", "error", "cancelled", "canceled":
		return "failed"
	case "expired", "timeout":
		return "expired"
	default:
		return s
	}
}
