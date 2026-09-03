"""What the emulated tool-calling block costs to send.

    python -m unittest discover -s tests -t .

Google drops a request that takes too long to produce its first byte, and
prompt size is what moves those odds. Measured on the deployment, 27 qwen tools
came to 56,016 characters of instruction — because _DESC_LIMIT only ever
applied to nested schema properties, so each tool's own description went
upstream verbatim.
"""

import unittest

from app.config import settings
from app.tools import (
    _TOOL_DESC_LIMIT,
    build_tool_instruction,
    instruction_costs,
)

LONG = "Runs a shell command on the host. " * 60  # ~2 KB, like qwen's
SHORT = "Reads a file."


def tool(name, desc, params=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": params or {"type": "object", "properties": {}},
        },
    }


TOOLS = [tool("run_shell_command", LONG), tool("read_file", SHORT)]


class DescriptionCap(unittest.TestCase):
    def test_a_long_description_is_truncated(self):
        out = build_tool_instruction(TOOLS, None)
        self.assertNotIn(LONG, out)
        self.assertIn(LONG[:200], out, "the opening sentence must survive")
        self.assertIn("…", out)

    def test_a_short_description_is_untouched(self):
        out = build_tool_instruction(TOOLS, None)
        self.assertIn(f"- read_file: {SHORT}", out)

    def test_the_cap_is_a_real_saving(self):
        capped = len(build_tool_instruction(TOOLS, None, _TOOL_DESC_LIMIT))
        uncapped = len(build_tool_instruction(TOOLS, None, 10**9))
        self.assertLess(capped, uncapped)
        self.assertGreater(uncapped - capped, 1000)

    def test_the_shipped_limit_is_what_the_setting_says(self):
        by_setting = build_tool_instruction(TOOLS, None, settings.tool_desc_limit)
        self.assertEqual(by_setting, build_tool_instruction(TOOLS, None))

    def test_nested_property_descriptions_are_still_capped_separately(self):
        """The tool cap must not have replaced _DESC_LIMIT."""
        nested = tool(
            "x",
            SHORT,
            {"type": "object", "properties": {"cmd": {"description": LONG}}},
        )
        out = build_tool_instruction([nested], None)
        self.assertNotIn(LONG[:400], out, "a property description is capped at 200")


class Costs(unittest.TestCase):
    """The logged split must not drift from what is actually sent."""

    def test_the_totals_match_the_built_instruction(self):
        desc, schema, _ = instruction_costs(TOOLS)
        boilerplate = len(build_tool_instruction([], None))
        full = len(build_tool_instruction(TOOLS, None))
        self.assertEqual(full - boilerplate, desc + schema)

    def test_tools_are_ranked_largest_first(self):
        _, _, per_tool = instruction_costs(TOOLS)
        self.assertEqual([name for name, _ in per_tool],
                         ["run_shell_command", "read_file"])

    def test_no_tools_costs_nothing(self):
        self.assertEqual(instruction_costs([]), (0, 0, []))
        self.assertEqual(instruction_costs(None), (0, 0, []))

    def test_the_cap_shows_up_in_the_numbers(self):
        capped, _, _ = instruction_costs(TOOLS, _TOOL_DESC_LIMIT)
        uncapped, _, _ = instruction_costs(TOOLS, 10**9)
        self.assertLess(capped, uncapped)


if __name__ == "__main__":
    unittest.main()
