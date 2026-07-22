"""OpenAI-compatible request/response schemas and helpers."""

import base64
import binascii
import json
import mimetypes
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Tuple

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict

from .config import settings


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
    # Function calling (Gemini-only, prompt-emulated). Standard OpenAI shapes.
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Any] = None


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
    # ComfyUI-Manager: install missing nodes / download missing models (from the
    # supplied full-graph `workflow`'s node types + top-level `models` array)
    # before generating. Defaults to settings.comfy_auto_provision.
    auto_provision: Optional[bool] = None
    # Ephemeral: don't persist this job on the ComfyUI box (temp output + delete
    # history). Defaults to settings.comfy_ephemeral.
    ephemeral: Optional[bool] = None


class ComfyProvisionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    # Full-graph workflow (may carry a top-level `models` array of
    # {name, url, directory, hash}); node class_types + models are what the
    # Manager installs. Also accepts API-format (nodes detected, no models).
    workflow: dict
    # Wait for the install queue to drain before returning; false -> start and
    # return immediately (poll GET /v1/comfy/provision/status).
    wait: bool = True
    # Override settings.comfy_provision_timeout (seconds) for the wait.
    timeout: Optional[float] = None


class ComfyAnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    # API-format workflow ({"<id>": {"class_type", "inputs"}}). We report which
    # node inputs take an uploaded file (LoadImage etc.).
    workflow: dict


class ComfyUploadRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    # data: URI or http(s) URL of the image/mask to import.
    image: str
    # Target filename in ComfyUI; a unique myrouter_<hex><ext> is generated
    # when omitted.
    filename: Optional[str] = None
    # Upload into the temp dir (auto-cleaned, not the permanent input gallery);
    # the returned `ref` carries the [temp] annotation. Default: COMFY_EPHEMERAL.
    ephemeral: Optional[bool] = None
    # Treat as an inpaint mask -> /upload/mask, compositing into original_ref's
    # alpha. `original_ref` = a prior upload's {filename, subfolder, type}.
    mask: bool = False
    original_ref: Optional[dict] = None


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


_MIME_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/heic": ".heic",
    "image/heif": ".heif",
    "image/bmp": ".bmp",
}


def _ext_for_mime(mime: str) -> str:
    mime = (mime or "").split(";")[0].strip().lower()
    return _MIME_EXT.get(mime) or mimetypes.guess_extension(mime) or ".bin"


def decode_image_source(src: str) -> Tuple[Optional[bytes], str]:
    """Decode a single image source string to (bytes, ext).

    A `data:` URI is decoded here → (bytes, ext); an http(s) URL returns
    (None, ext) so the caller fetches the bytes where the shared httpx client
    lives. Anything else → 400.
    """
    src = src or ""
    if src.startswith("data:"):
        header, _, b64 = src[len("data:") :].partition(",")
        mime = header.split(";")[0]
        try:
            data = base64.b64decode(b64, validate=False)
        except (binascii.Error, ValueError):
            raise openai_error(400, "Invalid base64 image data.")
        return data, _ext_for_mime(mime)
    if src.startswith(("http://", "https://")):
        mime = mimetypes.guess_type(src)[0] or "image/jpeg"
        return None, _ext_for_mime(mime)
    raise openai_error(400, "image must be a data: URI or an http(s) URL.")


def extract_images(content: Any) -> List[Dict[str, Any]]:
    """Pull OpenAI `image_url` parts from a message's content.

    Returns image sources: `{"data": bytes, "ext": str}` for `data:` URIs
    (decoded here), or `{"url": str}` for http(s) URLs (fetched later in the
    request path where the shared httpx client lives). Non-image content is
    ignored.
    """
    if not isinstance(content, list):
        return []
    images: List[Dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "image_url":
            continue
        url = (part.get("image_url") or {}).get("url", "")
        if not isinstance(url, str) or not url:
            continue
        if url.startswith("data:"):
            header, _, b64 = url[len("data:") :].partition(",")
            mime = header.split(";")[0]
            try:
                data = base64.b64decode(b64, validate=False)
            except (binascii.Error, ValueError):
                raise openai_error(400, "Invalid base64 image data in image_url.")
            images.append({"data": data, "ext": _ext_for_mime(mime)})
        elif url.startswith(("http://", "https://")):
            images.append({"url": url})
    return images


def collect_images(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """Images from the last user message (the current turn)."""
    for message in reversed(messages):
        if message.role == "user":
            return extract_images(message.content)
    return []


def flatten_messages(messages: List[ChatMessage]) -> str:
    """Collapse an OpenAI message array into a single prompt string.

    Renders the function-calling round-trip so the model keeps context:
    assistant `tool_calls` and `role:"tool"` results become readable lines.
    """
    lines = []
    for message in messages:
        role = message.role
        tool_calls = getattr(message, "tool_calls", None)
        if role == "assistant" and tool_calls:
            rendered = []
            for tc in tool_calls:
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                rendered.append(f"{fn.get('name', '?')}({fn.get('arguments', '')})")
            lines.append(f"assistant: [called tools: {'; '.join(rendered)}]")
            text = _content_to_text(message.content).strip()
            if text:
                lines.append(f"assistant: {text}")
            continue
        if role == "tool":
            tcid = getattr(message, "tool_call_id", None)
            text = _content_to_text(message.content).strip()
            label = f"tool result[{tcid}]" if tcid else "tool result"
            lines.append(f"{label}: {text}")
            continue
        text = _content_to_text(message.content).strip()
        if text:
            lines.append(f"{role}: {text}")
    return "\n\n".join(lines)


def build_chat_response(
    model: str,
    prompt: str,
    answer: str,
    tool_calls: Optional[List[dict]] = None,
    images: Optional[List[dict]] = None,
) -> dict:
    message: Dict[str, Any] = {"role": "assistant"}
    if tool_calls:
        message["content"] = None
        message["tool_calls"] = tool_calls
        finish = "tool_calls"
        completion_chars = len(json.dumps(tool_calls))
    else:
        message["content"] = answer
        finish = "stop"
        completion_chars = len(answer)
    # Extra media the Gemini backend returned (web + generated images). Not a
    # standard OpenAI field — ignored by plain SDK clients, rendered by the
    # playground.
    if images:
        message["images"] = images
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        # The upstream services expose no token counts; estimate ~4 chars/token.
        "usage": {
            "prompt_tokens": max(1, len(prompt) // 4),
            "completion_tokens": max(1, completion_chars // 4),
            "total_tokens": max(2, len(prompt) // 4 + completion_chars // 4),
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
    if settings.sse_include_done:
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
