package moark

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/o4openai/internal/model"
	"github.com/o4openai/pkg/utils"
)

// ============================================================
// Chat Completions adapter - OpenAI-compatible passthrough
//
// Moark's /v1/chat/completions accepts the standard OpenAI request
// shape (model, messages, temperature, top_p, tools, etc.) and returns
// the standard OpenAI response shape. So the conversion surface is
// almost zero:
//
//   - Resolve the external model name to a Moark model name.
//   - For multimodal messages, base64 image_url data URIs are converted
//     to temp URLs so Moark can fetch them (it doesn't accept raw base64).
//   - Stream responses use the same SSE format as OpenAI, so we just
//     return the raw response body.
// ============================================================

// ChatCompletion sends a non-streaming chat request.
func (p *Provider) ChatCompletion(ctx context.Context, req *model.ChatCompletionRequest) (*model.ChatCompletionResponse, error) {
	body, err := p.buildChatRequest(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to build chat request: %w", err)
	}

	resp, err := p.doJSON(ctx, "POST", "/chat/completions", body)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var out model.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("failed to decode Moark chat response: %w", err)
	}
	return &out, nil
}

// ChatCompletionStream opens an SSE stream and returns the raw body.
// The Moark SSE format matches OpenAI's (data: {json}\n\n + data: [DONE]),
// so the handler can re-emit it verbatim.
func (p *Provider) ChatCompletionStream(ctx context.Context, req *model.ChatCompletionRequest) (io.ReadCloser, error) {
	body, err := p.buildChatRequest(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to build chat request: %w", err)
	}
	// Moark streams when stream=true is set on the body.
	body["stream"] = true

	resp, err := p.doJSON(ctx, "POST", "/chat/completions", body)
	if err != nil {
		return nil, err
	}
	return resp.Body, nil
}

// buildChatRequest marshals the OpenAI-shaped request into a generic map
// suitable for JSON encoding. We use a map (not the typed struct) so that
// any provider-specific extensions (e.g. guided_json, guided_choice) that
// the Moark spec advertises but our internal model doesn't model are
// preserved when the client passes them in via Extra.
func (p *Provider) buildChatRequest(ctx context.Context, req *model.ChatCompletionRequest) (map[string]interface{}, error) {
	// Start from the request as JSON, then rewrite a few fields.
	// This preserves any fields the gateway's internal model doesn't
	// know about (since the request may have been deserialized from a
	// raw client body that included Moark-specific knobs).
	data, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	body := map[string]interface{}{}
	if err := json.Unmarshal(data, &body); err != nil {
		return nil, err
	}

	// Resolve the model name.
	body["model"] = p.resolveModel(req.Model)

	// Merge provider-specific fields from Extra (e.g. top_k, guided_json,
	// guided_choice) that the client passed but our internal model doesn't
	// model as typed fields.
	for k, v := range req.Extra {
		body[k] = v
	}

	// Convert base64 image_url data URIs in any multimodal messages to
	// temp URLs so Moark can fetch them.
	if msgs, ok := body["messages"].([]interface{}); ok {
		for _, raw := range msgs {
			m, ok := raw.(map[string]interface{})
			if !ok {
				continue
			}
			content, ok := m["content"].([]interface{})
			if !ok {
				continue
			}
			for _, rawPart := range content {
				part, ok := rawPart.(map[string]interface{})
				if !ok {
					continue
				}
				if part["type"] != "image_url" {
					continue
				}
				iu, ok := part["image_url"].(map[string]interface{})
				if !ok {
					continue
				}
				url, _ := iu["url"].(string)
				if url == "" || !utils.IsDataURL(url) {
					continue
				}
				if p.base64Handler == nil {
					p.logger.Warn("Base64 image received but Base64Handler is not configured; " +
						"Moark will likely reject the request.")
					continue
				}
				reqCtx := utils.RequestContextFromCtx(ctx)
				converted, _, err := p.base64Handler.ConvertDataURL(url, reqCtx)
				if err != nil {
					return nil, fmt.Errorf("failed to convert base64 image to temp URL: %w", err)
				}
				iu["url"] = converted
				part["image_url"] = iu
			}
		}
	}

	return body, nil
}
