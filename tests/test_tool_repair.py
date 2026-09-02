"""Tool-call parsing and the repair retry.

Run with the stdlib runner - MyRouter carries no test dependency:

    python -m unittest discover -s tests

The bug these pin down: an agentic client sends tools on every request, so
every reply went through the repair heuristic. That heuristic asked "is this
reply shorter than 200 characters?", which is true of most good answers, and
the repair prompt ("no prose, nothing outside the fence") converts a good
answer into a refusal. A greeting came back as "I cannot generate a
tool-calling JSON block…" after a second 16-second round trip.
"""

import unittest

from app.routes.chat import _needs_repair
from app.schemas import ChatCompletionRequest
from app.tools import looks_like_a_mangled_call, parse_tool_calls

NAMES = ["run_shell_command", "read_file"]

TOOLS = [
    {"type": "function", "function": {"name": name, "parameters": {}}}
    for name in NAMES
]


def request(tool_choice=None) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="gemini-3-pro",
        messages=[{"role": "user", "content": "alo?"}],
        tools=TOOLS,
        tool_choice=tool_choice,
    )


class NeedsRepair(unittest.TestCase):
    def test_a_short_conversational_reply_is_left_alone(self):
        """The reported case: 20 characters, no tools involved, no re-ask."""
        self.assertFalse(_needs_repair(request(), "Chào bạn! Tôi nghe đây.", NAMES))

    def test_ordinary_short_answers_are_left_alone(self):
        for text in ("Done.", "Disk usage is 41%.", "nginx is running.", "Yes."):
            with self.subTest(text=text):
                self.assertFalse(_needs_repair(request(), text, NAMES))

    def test_a_mangled_call_is_still_repaired(self):
        """The protection the length test was standing in for."""
        text = 'I will run run_shell_command with {"command": "df -h"'
        self.assertTrue(_needs_repair(request(), text, NAMES))

    def test_an_empty_reply_is_repaired(self):
        """The genuine stub the length rule was reaching for."""
        for text in ("", "   ", "\n\n"):
            with self.subTest(text=text):
                self.assertTrue(_needs_repair(request(), text, NAMES))

    def test_tool_choice_required_always_repairs(self):
        self.assertTrue(_needs_repair(request("required"), "Hello.", NAMES))

    def test_tool_choice_dict_always_repairs(self):
        choice = {"type": "function", "function": {"name": "read_file"}}
        self.assertTrue(_needs_repair(request(choice), "Hello.", NAMES))

    def test_a_tool_name_alone_is_not_evidence(self):
        """Talking about a tool is not attempting to call one."""
        text = "run_shell_command is how I would check that."
        self.assertFalse(looks_like_a_mangled_call(text, NAMES))

    def test_json_alone_is_not_evidence(self):
        self.assertFalse(looks_like_a_mangled_call('{"status": "ok"}', NAMES))


class QwenBracketNotation(unittest.TestCase):
    """qwen-code teaches `[tool_call: name {...}]`, not a fenced block."""

    def test_a_bracket_call_parses(self):
        text = '[tool_call: run_shell_command {"command": "df -h"}]'
        calls, cleaned = parse_tool_calls(text, NAMES)
        self.assertIsNotNone(calls)
        self.assertEqual(calls[0]["function"]["name"], "run_shell_command")
        self.assertEqual(calls[0]["function"]["arguments"], '{"command": "df -h"}')
        self.assertEqual(cleaned, "", "the notation must not leak into the chat")

    def test_prose_around_the_call_survives(self):
        text = 'Checking now. [tool_call: read_file {"path": "/etc/hosts"}] one moment.'
        calls, cleaned = parse_tool_calls(text, NAMES)
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(cleaned, "Checking now.  one moment.")

    def test_braces_inside_an_argument_do_not_end_the_span(self):
        text = '[tool_call: run_shell_command {"command": "awk \'{print $1}\' x"}]'
        calls, cleaned = parse_tool_calls(text, NAMES)
        self.assertIn("awk", calls[0]["function"]["arguments"])
        self.assertEqual(cleaned, "")

    def test_an_unknown_tool_name_is_ignored(self):
        text = '[tool_call: rm_minus_rf {"path": "/"}]'
        calls, _ = parse_tool_calls(text, NAMES)
        self.assertIsNone(calls)

    def test_a_truncated_call_is_not_a_call(self):
        text = '[tool_call: run_shell_command {"command": "df -h"'
        calls, cleaned = parse_tool_calls(text, NAMES)
        self.assertIsNone(calls)
        self.assertEqual(cleaned, text)


class FencedNotationStillWorks(unittest.TestCase):
    """Regressions on the path that already worked."""

    def test_a_fenced_block_parses(self):
        text = (
            "```tool_calls\n"
            '[{"name": "run_shell_command", "arguments": {"command": "uptime"}}]\n'
            "```"
        )
        calls, cleaned = parse_tool_calls(text, NAMES)
        self.assertEqual(calls[0]["function"]["name"], "run_shell_command")
        self.assertEqual(cleaned, "")

    def test_an_unfenced_json_array_parses(self):
        text = '[{"name": "read_file", "arguments": {"path": "/tmp/a"}}]'
        calls, _ = parse_tool_calls(text, NAMES)
        self.assertEqual(calls[0]["function"]["name"], "read_file")

    def test_a_plain_answer_is_not_a_call(self):
        calls, cleaned = parse_tool_calls("Disk usage is 41%.", NAMES)
        self.assertIsNone(calls)
        self.assertEqual(cleaned, "Disk usage is 41%.")


class RepairOutcome(unittest.TestCase):
    """The choice _run_tool_turn makes when a repair yields no calls.

    Mirrors the branch rather than running the turn: _run_tool_turn needs a live
    Gemini client. The rule is one line and it is the one that shipped the
    refusal, so it is worth pinning.
    """

    @staticmethod
    def resolve(original: str, repair: str) -> str:
        return repair if not original.strip() else original

    def test_a_refusal_never_replaces_a_good_answer(self):
        answer = "Chào bạn!"
        refusal = (
            "I cannot generate a tool-calling JSON block because those "
            "management functions are not available in this environment."
        )
        self.assertGreater(len(refusal), len(answer), "the old rule preferred this")
        self.assertEqual(self.resolve(answer, refusal), answer)

    def test_an_empty_original_takes_the_repair(self):
        self.assertEqual(self.resolve("   ", "Here you go."), "Here you go.")


if __name__ == "__main__":
    unittest.main()
