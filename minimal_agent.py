"""
Minimal agent loop — Anthropic SDK + one mock tool.

Run:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python minimal_agent.py
"""

import json
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()
MODEL = "claude-sonnet-4-6"

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a given city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Taipei'"},
            },
            "required": ["city"],
        },
    }
]


def execute_tool(name: str, args: dict) -> str:
    """Dispatch table for tool execution. Return a string result."""
    if name == "get_weather":
        # Mock — swap with a real API call when you're ready.
        return json.dumps({"city": args["city"], "temp_c": 22, "condition": "sunny"})
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
        print(f"turn: {turn}, messages:{messages}");
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        # Always append the assistant turn verbatim — the API requires it
        # because tool_use ids must be matched in the next user turn.
        messages.append({"role": "assistant", "content": response.content})
        print(f"turn: {turn}, messages:{messages}");

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
    answer = run_agent("What's the weather like in Taipei right now?")
    print("\n=== Final answer ===")
    print(answer)
