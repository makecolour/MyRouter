"""Prompt-emulated OpenAI function calling for the Gemini backend.

The reverse-engineered Gemini web API has no native tool API, so we inject
the tool schemas into the prompt with a strict output contract, then parse
the model's reply back into OpenAI `tool_calls`. Works well with capable
models (gemini-3-pro); inherently best-effort.
"""

import json
import re
import uuid
from typing import Any, List, Optional, Tuple

_FENCE_RE = re.compile(
    r"```(?:tool_calls|json)?\s*(\[.*?\]|\{.*?\})\s*```", re.DOTALL | re.IGNORECASE
)


def tool_names(tools: Optional[List[dict]]) -> List[str]:
    names = []
    for tool in tools or []:
        if isinstance(tool, dict):
            fn = tool.get("function") or {}
            name = fn.get("name")
            if name:
                names.append(name)
    return names


def tools_requested(tools: Optional[List[dict]], tool_choice: Any) -> bool:
    """True when the request wants tool calling (and it isn't disabled)."""
    if not tools:
        return False
    if isinstance(tool_choice, str) and tool_choice.strip().lower() == "none":
        return False
    return True


def build_tool_instruction(tools: List[dict], tool_choice: Any) -> str:
    """A protocol block describing the tools + the exact output contract."""
    lines = [
        "You can call tools. When you decide to use one or more tools, reply "
        "with ONLY a fenced code block labelled tool_calls containing a JSON "
        "array — no prose before or after it. Each element must be "
        '{"name": <tool name>, "arguments": <object matching the tool\'s '
        "parameters>}. Example:",
        "```tool_calls",
        '[{"name": "get_weather", "arguments": {"city": "Hanoi"}}]',
        "```",
        "If no tool is needed, answer the user normally with NO tool_calls "
        "block.",
        "",
        "Available tools:",
    ]
    for tool in tools:
        fn = tool.get("function") or {} if isinstance(tool, dict) else {}
        name = fn.get("name", "?")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        lines.append(f"- {name}: {desc}")
        lines.append(f"  parameters (JSON Schema): {json.dumps(params, ensure_ascii=False)}")

    # Forcing semantics from tool_choice.
    if isinstance(tool_choice, str) and tool_choice.strip().lower() == "required":
        lines.append("")
        lines.append("You MUST call at least one tool for this turn.")
    elif isinstance(tool_choice, dict):
        forced = (tool_choice.get("function") or {}).get("name")
        if forced:
            lines.append("")
            lines.append(f"You MUST call the tool `{forced}` for this turn.")

    return "\n".join(lines)


def _coerce_calls(parsed: Any) -> List[dict]:
    if isinstance(parsed, dict):
        # Either a single {name, arguments} or {"tool_calls": [...]}.
        if "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
            return [c for c in parsed["tool_calls"] if isinstance(c, dict)]
        return [parsed]
    if isinstance(parsed, list):
        return [c for c in parsed if isinstance(c, dict)]
    return []


def parse_tool_calls(
    text: str, valid_names: List[str]
) -> Tuple[Optional[List[dict]], str]:
    """Extract OpenAI `tool_calls` from the model's reply.

    Returns (tool_calls | None, text_without_the_block). Only calls whose name
    is in `valid_names` are kept. None means the model answered normally.
    """
    if not text:
        return None, text

    candidates = []
    cleaned = text
    match = _FENCE_RE.search(text)
    if match:
        candidates.append(match.group(1))
        cleaned = (text[: match.start()] + text[match.end() :]).strip()
    else:
        stripped = text.strip()
        if stripped[:1] in ("[", "{"):
            candidates.append(stripped)

    allowed = set(valid_names)
    for raw in candidates:
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue
        calls = []
        for call in _coerce_calls(parsed):
            name = call.get("name")
            if not name or (allowed and name not in allowed):
                continue
            args = call.get("arguments", {})
            if isinstance(args, str):
                # Some models already stringify arguments — keep as-is if it
                # parses, else wrap.
                try:
                    json.loads(args)
                    args_str = args
                except (json.JSONDecodeError, ValueError):
                    args_str = json.dumps({"_raw": args})
            else:
                args_str = json.dumps(args, ensure_ascii=False)
            calls.append(
                {
                    "id": "call_" + uuid.uuid4().hex[:24],
                    "type": "function",
                    "function": {"name": name, "arguments": args_str},
                }
            )
        if calls:
            return calls, cleaned

    return None, text
