# O4OpenAI API Gateway
# Multi-stage build for a small final image

FROM golang:1.22-alpine AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
ARG TARGETOS=linux
ARG TARGETARCH=amd64
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
    go build -ldflags="-s -w" -o /out/o4openai ./cmd/server/

FROM alpine:3.19
RUN apk add --no-cache ca-certificates tzdata && \
    adduser -D -u 1000 app
WORKDIR /app
COPY --from=builder /out/o4openai /app/o4openai
COPY chat.html /app/chat.html
COPY config.yaml /app/config.yaml
USER app
EXPOSE 1241
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD wget -qO- http://localhost:1241/health || exit 1
ENTRYPOINT ["/app/o4openai"]
CMD ["-config", "/app/config.yaml"]
