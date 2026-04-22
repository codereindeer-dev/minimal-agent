"""
Minimal agent loop — Anthropic SDK + one mock tool.

Run:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python minimal_agent.py
"""

import json
import subprocess
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()
MODEL = "claude-sonnet-4-6"

TOOLS = [
    {
        "name": "run_shell",
        "description": "Run a shell command and return stdout, stderr, and exit code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a text file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write text content to a file, overwriting any existing content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write"},
                "content": {"type": "string", "description": "Text content to write"},
            },
            "required": ["path", "content"],
        },
    },
]


def execute_tool(name: str, args: dict) -> str:
    """Dispatch table for tool execution. Return a string result."""
    if name == "run_shell":
        proc = subprocess.run(
            args["command"], shell=True, capture_output=True, text=True, timeout=30
        )
        return json.dumps({
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        })
    if name == "read_file":
        try:
            return Path(args["path"]).read_text(encoding="utf-8")
        except Exception as e:
            return f"ERROR reading {args['path']}: {e}"
    if name == "write_file":
        try:
            Path(args["path"]).write_text(args["content"], encoding="utf-8")
            return f"Wrote {len(args['content'])} chars to {args['path']}"
        except Exception as e:
            return f"ERROR writing {args['path']}: {e}"
    return f"ERROR: unknown tool {name}"


def run_agent(user_message: str, max_turns: int = 10) -> str:
    """
    The agent loop:
      1. Send messages to Claude.
      2. If Claude returns tool_use, execute tools and append tool_result.
      3. Repeat until Claude stops requesting tools (stop_reason == "end_turn").
    """
    messages = [{"role": "user", "content": user_message}]

    for turn in range(max_turns):
        #print(f"turn: {turn}, messages:{messages}");
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        # Always append the assistant turn verbatim — the API requires it
        # because tool_use ids must be matched in the next user turn.
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Final text answer — collect and return.
            return "".join(
                block.text for block in response.content if block.type == "text"
            )

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  [tool] {block.name}({block.input})")
                    result = execute_tool(block.name, block.input)
                    print(f"  [result] {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})
            continue

        # Any other stop_reason (max_tokens, pause_turn, refusal, ...) — bail out.
        raise RuntimeError(f"Unexpected stop_reason: {response.stop_reason}")

    raise RuntimeError(f"Agent exceeded {max_turns} turns")


if __name__ == "__main__":
    answer = run_agent(
        "List the Python files in the current directory, then summarize what minimal_agent.py does."
    )
    print("\n=== Final answer ===")
    print(answer)
