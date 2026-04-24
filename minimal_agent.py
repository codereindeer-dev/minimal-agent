"""
Minimal agent loop — Anthropic SDK + native tools + MCP server, multi-turn.

Run:
    pip install anthropic python-dotenv mcp mcp-server-fetch
    # put ANTHROPIC_API_KEY in .env
    python minimal_agent.py
"""

import asyncio
import json
import subprocess
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()
MODEL = "claude-sonnet-4-6"

NATIVE_TOOLS = [
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


def execute_native_tool(name: str, args: dict) -> str:
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
    return f"ERROR: unknown native tool {name}"


async def call_mcp_tool(session: ClientSession, name: str, args: dict) -> str:
    """Call a tool on the MCP server; flatten its content blocks into a string."""
    result = await session.call_tool(name, args)
    parts = []
    for block in result.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
        else:
            parts.append(str(block))
    return "\n".join(parts) if parts else ""


class Agent:
    """Multi-turn agent. Keeps message history across chat() calls."""

    def __init__(
        self,
        client: Anthropic,
        session: ClientSession,
        tools: list,
        mcp_tool_names: set,
        model: str = MODEL,
        max_turns: int = 10,
    ):
        self.client = client
        self.session = session
        self.tools = tools
        self.mcp_tool_names = mcp_tool_names
        self.model = model
        self.max_turns = max_turns
        self.messages: list = []

    async def _dispatch_tool(self, name: str, args: dict) -> str:
        if name in self.mcp_tool_names:
            return await call_mcp_tool(self.session, name, args)
        return execute_native_tool(name, args)

    async def chat(self, user_message: str) -> str:
        """Send one user message, run the tool-use loop, return the final reply."""
        self.messages.append({"role": "user", "content": user_message})

        for _ in range(self.max_turns):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                tools=self.tools,
                messages=self.messages,
            )
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                return "".join(
                    b.text for b in response.content if b.type == "text"
                )

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        source = "mcp" if block.name in self.mcp_tool_names else "native"
                        print(f"  [{source}] {block.name}({block.input})")
                        result = await self._dispatch_tool(block.name, block.input)
                        preview = result[:200] + ("..." if len(result) > 200 else "")
                        print(f"  [result] {preview}")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                self.messages.append({"role": "user", "content": tool_results})
                continue

            raise RuntimeError(f"Unexpected stop_reason: {response.stop_reason}")

        raise RuntimeError(f"Agent exceeded {self.max_turns} turns")

    def reset(self):
        self.messages = []


async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "mcp_server_fetch"],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listing = await session.list_tools()
            mcp_tools = [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.inputSchema,
                }
                for t in listing.tools
            ]
            mcp_tool_names = {t["name"] for t in mcp_tools}
            all_tools = NATIVE_TOOLS + mcp_tools

            print(f"[init] native tools: {[t['name'] for t in NATIVE_TOOLS]}")
            print(f"[init] mcp tools:    {sorted(mcp_tool_names)}")
            print("[hint] type /reset to clear history, /exit or Ctrl-D to quit.\n")

            agent = Agent(
                client=Anthropic(),
                session=session,
                tools=all_tools,
                mcp_tool_names=mcp_tool_names,
            )

            while True:
                try:
                    user_in = (await asyncio.to_thread(input, "you> ")).strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not user_in:
                    continue
                if user_in in ("/exit", "/quit"):
                    break
                if user_in == "/reset":
                    agent.reset()
                    print("[history cleared]\n")
                    continue

                reply = await agent.chat(user_in)
                print(f"\nclaude> {reply}\n")


if __name__ == "__main__":
    asyncio.run(main())
