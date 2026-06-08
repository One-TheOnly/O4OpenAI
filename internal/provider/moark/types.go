package moark

// ============================================================
// Moark (模力方舟) specific types - internal to the Moark provider.
//
// The Moark API is largely OpenAI-compatible, so most request and
// response types just decode into the standard model.OpenAI* structs.
// This file holds only the small set of types that are Moark-specific:
//   - error envelope
//   - async task envelope (used for video generation and image polling)
// ============================================================

// MoarkErrorResponse is the error envelope returned by Moark on 4xx/5xx.
type MoarkErrorResponse struct {
	Error MoarkError `json:"error"`
}

// MoarkError is the body of a Moark API error.
type MoarkError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Type    string `json:"type"`
}

// MoarkTaskResponse is the response returned by async task endpoints
// (e.g. POST /v1/async/videos/generations and GET /v1/task/{task_id}).
//
// The "output" field is a free-form object whose shape depends on the
// task type. For video generation it typically contains a "url" string
// pointing at the generated video. We decode it loosely into a map and
// pull out the URL at conversion time.
type MoarkTaskResponse struct {
	TaskID      string                 `json:"task_id"`
	Output      map[string]interface{} `json:"output"`
	Status      string                 `json:"status"`
	CreatedAt   int64                  `json:"created_at"`
	StartedAt   int64                  `json:"started_at"`
	CompletedAt int64                  `json:"completed_at"`
	ExpiresAt   int64                  `json:"expires_at"`
	Price       float64                `json:"price"`
	Currency    string                 `json:"currency"`
	URLs        *MoarkTaskURLs         `json:"urls,omitempty"`
	Error       *MoarkError            `json:"error,omitempty"`
}

// MoarkTaskURLs holds the helper URLs returned alongside a task.
type MoarkTaskURLs struct {
	Get    string `json:"get"`
	Cancel string `json:"cancel"`
}
