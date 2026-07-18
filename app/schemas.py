"""OpenAI-compatible request/response schemas and helpers."""

import json
import time
import uuid
from typing import Any, Iterator, List, Optional

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict


def openai_error(
    status_code: int,
    message: str,
    err_type: str = "invalid_request_error",
    code: Optional[str] = None,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"error": {"message": message, "type": err_type, "code": code}},
    )


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = None  # str, or list of content parts, per OpenAI spec


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # tolerate temperature, top_p, …

    model: str
    messages: List[ChatMessage]
    stream: bool = False
    # Gemini-only extension: "new" starts a server-side conversation, an
    # existing id continues it (only the last user message is sent). Absent ->
    # stateless (history flattened into one prompt).
    conversation_id: Optional[str] = None


class ImageGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt: str
    n: int = 1
    size: str = "1024x1024"
    # Name of a comfy_instances row; omitted -> first enabled instance.
    model: Optional[str] = None
    # Optional ComfyUI API-format workflow. Placeholders "{prompt}",
    # "{negative_prompt}", "{seed}", "{width}", "{height}" are substituted.
    workflow: Optional[dict] = None
    # Deep-customization knobs for the built-in workflow (defaults from
    # settings when omitted). Discover valid values per instance via
    # GET /v1/comfy/{instance}/info.
    negative_prompt: Optional[str] = None
    checkpoint: Optional[str] = None
    steps: Optional[int] = None
    cfg: Optional[float] = None
    sampler: Optional[str] = None
    scheduler: Optional[str] = None
    denoise: Optional[float] = None
    # Fixed seed for reproducibility; with n > 1 it increments per image.
    seed: Optional[int] = None
    # Delivery mode (9Router "Output Format"): url (default) | b64_json |
    # binary. Both OpenAI-style field names are accepted; response_format
    # wins when both are set.
    response_format: Optional[str] = None
    output_format: Optional[str] = None


def _content_to_text(content: Any) -> str:
    """Flatten OpenAI message content (string or content-part list) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def last_user_message(messages: List[ChatMessage]) -> Optional[str]:
    """The newest user message — what gets sent in conversation mode."""
    for message in reversed(messages):
        if message.role == "user":
            text = _content_to_text(message.content).strip()
            if text:
                return text
    return None


def flatten_messages(messages: List[ChatMessage]) -> str:
    """Collapse an OpenAI message array into a single prompt string."""
    lines = []
    for message in messages:
        text = _content_to_text(message.content).strip()
        if text:
            lines.append(f"{message.role}: {text}")
    return "\n\n".join(lines)


def build_chat_response(model: str, prompt: str, answer: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        # The upstream services expose no token counts; estimate ~4 chars/token.
        "usage": {
            "prompt_tokens": max(1, len(prompt) // 4),
            "completion_tokens": max(1, len(answer) // 4),
            "total_tokens": max(2, len(prompt) // 4 + len(answer) // 4),
        },
    }


def sse_chunks(
    model: str,
    prompt: str,
    answer: str,
    conversation_id: Optional[str] = None,
) -> Iterator[str]:
    """Single-answer SSE stream (NotebookLM ask() is not streamable).

    Emits the full answer in one content chunk, a finish chunk, then a usage
    chunk (choices:[]) so OpenAI-compatible routers can count tokens.
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def chunk(
        delta: dict,
        finish_reason: Optional[str] = None,
        usage: Optional[dict] = None,
        choices: bool = True,
    ) -> str:
        payload = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": (
                [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
                if choices
                else []
            ),
        }
        if usage is not None:
            payload["usage"] = usage
        if conversation_id:
            payload["conversation_id"] = conversation_id
        return f"data: {json.dumps(payload)}\n\n"

    usage = {
        "prompt_tokens": max(1, len(prompt) // 4),
        "completion_tokens": max(1, len(answer) // 4),
        "total_tokens": max(2, len(prompt) // 4 + len(answer) // 4),
    }
    yield chunk({"role": "assistant", "content": ""})
    yield chunk({"content": answer})
    # Usage ON the finish chunk — routers that stop at finish_reason would
    # drop a trailing usage-only chunk (see chat._gemini_stream_response).
    yield chunk({}, finish_reason="stop", usage=usage)
    yield "data: [DONE]\n\n"


def extract_text(result: Any) -> str:
    """Pull the answer text out of an upstream client's response object."""
    if isinstance(result, str):
        return result
    for attr in ("text", "answer", "content", "response"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value:
            return value
    return str(result)
