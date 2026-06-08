#!/bin/bash
# O4OpenAI Linux deployment helper
# Usage: ./scripts/deploy.sh [arch]
#   arch: amd64 (default) | arm64

set -euo pipefail

ARCH="${1:-amd64}"
BINARY="dist/o4openai-linux-${ARCH}"

if [ ! -f "$BINARY" ]; then
    echo "Building for linux/${ARCH}..."
    mkdir -p dist
    CGO_ENABLED=0 GOOS=linux GOARCH="${ARCH}" \
        go build -ldflags="-s -w" -o "${BINARY}" ./cmd/server/
fi

# Sanity check
file "$BINARY"
echo ""
echo "Binary: $BINARY"
echo "Size:   $(du -h "$BINARY" | cut -f1)"
echo "SHA256: $(shasum -a 256 "$BINARY" | cut -d' ' -f1)"
echo ""
echo "Note: Copy chat.html alongside the binary for ChatBot UI (/chatbot)"
echo ""

# Optional: install to /usr/local/bin
if [ "${INSTALL:-0}" = "1" ]; then
    sudo install -m 0755 "$BINARY" /usr/local/bin/o4openai
    echo "Installed to /usr/local/bin/o4openai"
fi
