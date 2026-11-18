"""Request/image ceilings taken from upstream limits, not Bridge policy.

MAX_REQUEST_BYTES: Claude Code's raw API request ceiling (32 MiB), the smallest
request ceiling on any inference path. Codex/OpenAI (512 MB), the Claude API
(128 MB) and Cloudflare Workers Free/Pro request bodies (100 MB) are larger.
CLAUDE_MAX_IMAGE_BASE64: the Claude API per-image ceiling, measured on the
base64 string (10 MB). Codex resizes data-URL images itself and only rejects
above its own decoder guard (>= 79 MiB), so the Codex path has no separate
per-image ceiling below MAX_REQUEST_BYTES. See docs/DECISIONS.md D-50.
"""
MAX_REQUEST_BYTES = 32 * 1024 * 1024
CLAUDE_MAX_IMAGE_BASE64 = 10 * 1024 * 1024
# A relay work carries the request body as a JSON string; quotes/backslashes
# escape to two bytes. The Worker never dispatches a larger work to this agent.
MAX_RELAY_WORK_BYTES = 2 * MAX_REQUEST_BYTES + 65536
