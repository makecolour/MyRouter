"""Prompt-emulated OpenAI function calling for the Gemini backend.

The reverse-engineered Gemini web API has no native tool API, so we inject
the tool schemas into the prompt with a strict output contract, then parse
the model's reply back into OpenAI `tool_calls`. Works well with capable
models (gemini-3-pro); inherently best-effort.
"""

import json
import re
import uuid
from typing import Any, Iterator, List, Optional, Tuple

_FENCE_RE = re.compile(
    r"```(?:tool_calls?|function_calls?|json)?\s*(\[.*?\]|\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)
# Models that ignore the fence still tend to emit the JSON somewhere in the
# reply, so a bracket scan is the last resort before giving up.
_JSON_START_RE = re.compile(r"[\[{]")
# qwen-code teaches its models a bracket notation rather than a fenced block:
#     [tool_call: run_shell_command {"command": "df -h"}]
# It opens with "[", so the unfenced scan below grabs it and json.loads chokes
# on `tool_call:`. Unrecognised, the call leaks into the chat as prose and
# nothing ever runs. Matching it here fixes every qwen client at once.
_BRACKET_CALL_RE = re.compile(
    r"\[\s*(?:tool_calls?|called\s+tools?|function_calls?)\s*:\s*"
    r"([A-Za-z_][\w.-]*)\s*",
    re.IGNORECASE,
)
# OpenAI says "arguments"; emulated models reach for whatever the schema called
# it. All of these mean the same thing.
_ARG_ALIASES = ("arguments", "args", "parameters", "input", "params")


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


# Schema keys that carry no information the model needs in order to emit a
# valid call. An agentic client sends dozens of tools at once, and dumping every
# full schema inflates the prompt to the point where Google silently aborts the
# generation — so these go, and prose is trimmed. Names, types, `required` and
# `enum` are what actually constrain the output.
_SCHEMA_NOISE = frozenset(
    {"$schema", "$id", "title", "additionalProperties", "default", "examples"}
)
_DESC_LIMIT = 200
# Below this nesting depth, a property's prose is dead weight: the model needs
# the shape of a nested object, not an essay about each leaf.
_DESC_MAX_DEPTH = 2


def _compact_schema(node: Any, depth: int = 0) -> Any:
    """Strip a JSON Schema down to what constrains a generated call."""
    if isinstance(node, list):
        return [_compact_schema(item, depth) for item in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for key, value in node.items():
        if key in _SCHEMA_NOISE:
            continue
        if key == "description":
            if depth > _DESC_MAX_DEPTH or not isinstance(value, str):
                continue
            text = value.strip()
            if len(text) > _DESC_LIMIT:
                text = text[:_DESC_LIMIT].rstrip() + "…"
            if text:
                out[key] = text
            continue
        # "properties" holds tool-defined names, so its keys are data, not
        # schema keywords — descend without treating them as such.
        out[key] = _compact_schema(value, depth + 1)
    return out


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
        params = _compact_schema(fn.get("parameters", {}))
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

    # This block is appended AFTER the conversation, so the contract is the last
    # thing read before generation rather than being buried tens of KB above it.
    lines.append("")
    lines.append(
        "Reminder: to call a tool, output ONLY the ```tool_calls fenced JSON "
        "array and nothing else."
    )

    return "\n".join(lines)


def build_repair_instruction(tools: List[dict], tool_choice: Any) -> str:
    """A terse re-ask for when the model ignored the contract entirely."""
    names = ", ".join(tool_names(tools)) or "the available tools"
    forced = ""
    if isinstance(tool_choice, dict):
        forced_name = (tool_choice.get("function") or {}).get("name")
        if forced_name:
            forced = f" You must call `{forced_name}`."
    return (
        "Your previous reply did not follow the tool-calling format." + forced + "\n"
        "Reply again with ONLY a fenced code block labelled tool_calls holding a "
        'JSON array of {"name": ..., "arguments": {...}} objects — no prose, no '
        "explanation, nothing outside the fence. Valid tool names: " + names + ".\n"
        "If genuinely no tool applies, answer the user's question directly instead."
    )


def _coerce_calls(parsed: Any) -> List[dict]:
    if isinstance(parsed, dict):
        # Either a single {name, arguments} or {"tool_calls": [...]}.
        for key in ("tool_calls", "function_calls", "calls"):
            if isinstance(parsed.get(key), list):
                return [c for c in parsed[key] if isinstance(c, dict)]
        return [parsed]
    if isinstance(parsed, list):
        return [c for c in parsed if isinstance(c, dict)]
    return []


def _call_arguments(call: dict) -> Any:
    # A model echoing OpenAI's own wire shape nests both name and arguments
    # under "function"; look there too before falling back to an empty object.
    sources = [call]
    if isinstance(call.get("function"), dict):
        sources.append(call["function"])
    for source in sources:
        for alias in _ARG_ALIASES:
            if alias in source:
                return source[alias]
    return {}


def _balanced_end(text: str, start: int) -> Optional[int]:
    """Index just past the balanced JSON value opening at `start`, else None.

    Skips string literals so a brace inside a tool argument can't end the span
    early.
    """
    close = "]" if text[start] == "[" else "}"
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
            if depth == 0:
                return i + 1 if char == close else None
    return None


def _iter_json_spans(text: str) -> Iterator[Tuple[str, int, int]]:
    """Yield (raw, start, end) for every balanced JSON array/object in `text`.

    Used when the model dropped the fence.
    """
    for opener in _JSON_START_RE.finditer(text):
        start = opener.start()
        end = _balanced_end(text, start)
        if end is not None:
            yield text[start:end], start, end


def _iter_bracket_calls(text: str) -> Iterator[Tuple[str, int, int]]:
    """Yield qwen's `[tool_call: name {...}]` rewritten as ordinary call JSON.

    The rewrite is textual so the result flows through the same json.loads and
    name allowlist as every other candidate — a malformed argument object just
    fails to parse and the scan moves on.
    """
    for match in _BRACKET_CALL_RE.finditer(text):
        args_at = match.end()
        if args_at >= len(text) or text[args_at] != "{":
            continue
        args_end = _balanced_end(text, args_at)
        if args_end is None:
            continue
        raw = '{"name": %s, "arguments": %s}' % (
            json.dumps(match.group(1)),
            text[args_at:args_end],
        )
        # Swallow the closing bracket too, so the notation leaves nothing behind
        # in the chat.
        tail = text[args_end:]
        stripped = tail.lstrip()
        end = args_end
        if stripped.startswith("]"):
            end += len(tail) - len(stripped) + 1
        yield raw, match.start(), end


def looks_like_a_mangled_call(text: str, valid_names: List[str]) -> bool:
    """True when the reply reads like a tool call the parser could not read.

    A tool name sitting next to JSON punctuation is evidence the model tried and
    botched the contract. Anything else — however short — is an answer.
    """
    if not text or not _JSON_START_RE.search(text):
        return False
    return any(name and name in text for name in valid_names)


def parse_tool_calls(
    text: str, valid_names: List[str]
) -> Tuple[Optional[List[dict]], str]:
    """Extract OpenAI `tool_calls` from the model's reply.

    Returns (tool_calls | None, text_without_the_block). Only calls whose name
    is in `valid_names` are kept. None means the model answered normally.
    """
    if not text:
        return None, text

    # (raw JSON, span to cut from the reply if this is the one that parses).
    # Every fenced block is tried, not just the first: a model that narrates
    # before complying leaves a prose fence ahead of the real one.
    candidates: List[Tuple[str, int, int]] = [
        (match.group(1), match.start(), match.end())
        for match in _FENCE_RE.finditer(text)
    ]
    # qwen's bracket notation, ahead of the generic scan: that scan sees the
    # same "[" and would only fail to parse it.
    candidates.extend(_iter_bracket_calls(text))
    # Unfenced fallback: any balanced JSON value in the reply.
    candidates.extend(_iter_json_spans(text))

    allowed = set(valid_names)
    for raw, start, end in candidates:
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue
        calls = []
        for call in _coerce_calls(parsed):
            name = call.get("name") or (call.get("function") or {}).get("name")
            if not name or (allowed and name not in allowed):
                continue
            args = _call_arguments(call)
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
            return calls, (text[:start] + text[end:]).strip()

    return None, text
