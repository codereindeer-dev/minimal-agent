"""
Minimal agent loop — Anthropic SDK + native tools + MCP server, multi-turn.

Run:
    pip install anthropic python-dotenv mcp mcp-server-fetch
    # put ANTHROPIC_API_KEY in .env
    python minimal_agent.py
"""

import argparse
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
SESSIONS_DIR = Path("sessions")
SUMMARY_PREFIX = "[Earlier conversation summary]\n"

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


def _serialize_messages(messages: list) -> list:
    """Convert SDK content blocks (pydantic) into JSON-safe dicts."""
    out = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            out.append({"role": msg["role"], "content": content})
            continue
        blocks = [b.model_dump() if hasattr(b, "model_dump") else b for b in content]
        out.append({"role": msg["role"], "content": blocks})
    return out


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
        max_input_tokens: int = 100_000,
        keep_recent_turns: int = 5,
    ):
        self.client = client
        self.session = session
        self.tools = tools
        self.mcp_tool_names = mcp_tool_names
        self.model = model
        self.max_turns = max_turns
        self.max_input_tokens = max_input_tokens
        self.keep_recent_turns = keep_recent_turns
        self.messages: list = []
        self.last_input_tokens: int = 0  # from most recent response.usage

    @staticmethod
    def _describe_block(block) -> str:
        """Return a short label for one content block (SDK object or dict)."""
        get = (lambda k: block.get(k)) if isinstance(block, dict) else (lambda k: getattr(block, k, None))
        btype = get("type")
        if btype == "text":
            return "text"
        if btype == "tool_use":
            return f"tool_use:{get('name')}"
        if btype == "tool_result":
            return "tool_result"
        return btype or "?"

    def dump(self, max_text: int = 50) -> None:
        """Print a one-line-per-message summary of the conversation history."""
        if not self.messages:
            print("  (empty)")
            return
        for i, msg in enumerate(self.messages):
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, str):
                text = content.replace("\n", " ")
                if len(text) > max_text:
                    text = text[:max_text] + "..."
                desc = f'"{text}"'
            else:
                desc = "[" + " + ".join(self._describe_block(b) for b in content) + "]"
            print(f"  index {i}: {role:<9} {desc}")

    def count_tokens(self) -> int:
        """Ask the API how many input tokens our current messages+tools use."""
        if not self.messages:
            return 0
        result = self.client.messages.count_tokens(
            model=self.model,
            tools=self.tools,
            messages=self.messages,
        )
        return result.input_tokens

    def _has_summary_prefix(self) -> bool:
        """True if messages[0] is a rolled-summary placeholder."""
        if not self.messages:
            return False
        m = self.messages[0]
        return (
            m["role"] == "user"
            and isinstance(m["content"], str)
            and m["content"].startswith(SUMMARY_PREFIX)
        )

    def _turn_starts(self) -> list[int]:
        """Indices of messages that begin a turn (user role with string content)."""
        return [
            i for i, m in enumerate(self.messages)
            if m["role"] == "user" and isinstance(m["content"], str)
        ]

    async def _summarize_messages(self, messages_to_summarize: list,
                                  has_prior_summary: bool) -> str:
        """Ask the model for a single recap covering messages_to_summarize.
        If has_prior_summary, the first message in the list is itself a previous
        recap — the new summary should fold the new turns into the old one."""
        if not messages_to_summarize:
            return ""
        if has_prior_summary:
            instruction = (
                "Above you'll see an existing running summary plus newer turns. "
                "Produce ONE updated running summary that integrates the new turns "
                "into the old recap. Keep key facts, decisions, file paths, and any "
                "outstanding work. Plain text only, no preamble, no headings."
            )
        else:
            instruction = (
                "Summarize the conversation above into a concise recap I can use to "
                "continue the task. Include key facts, decisions, file paths, and any "
                "outstanding work. Plain text only, no preamble, no headings."
            )
        prompt = messages_to_summarize + [{"role": "user", "content": instruction}]
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=prompt,
        )
        return "".join(b.text for b in response.content if b.type == "text")

    async def _trim_to_budget(self) -> int:
        """Roll oldest turns into a summary message until under the input-token
        budget. Keeps the last `keep_recent_turns` turns verbatim. Returns the
        number of turns folded into the summary."""
        if not self.messages or self.count_tokens() <= self.max_input_tokens:
            return 0

        folded_total = 0
        # One pass is normally enough: after folding, the next iteration has
        # exactly keep_recent_turns real turns and excess becomes 0. The while
        # is here for safety, not repeated folding.
        while self.count_tokens() > self.max_input_tokens:
            starts = self._turn_starts()
            has_summary = self._has_summary_prefix()
            real_starts = starts[1:] if has_summary else starts
            excess = len(real_starts) - self.keep_recent_turns
            if excess <= 0:
                break  # can't trim further without dropping recent context
            cutoff = real_starts[excess]  # first kept turn
            to_summarize = self.messages[:cutoff]
            recent = self.messages[cutoff:]
            new_summary = await self._summarize_messages(
                to_summarize, has_prior_summary=has_summary
            )
            self.messages = [
                {"role": "user", "content": SUMMARY_PREFIX + new_summary},
                {"role": "assistant", "content": "Understood. Continuing from the summary."},
            ] + recent
            folded_total += excess
        return folded_total

    async def compact(self) -> str:
        """Manually summarize the entire history into a single recap message."""
        if not self.messages:
            return "(nothing to compact)"
        has_summary = self._has_summary_prefix()
        summary = await self._summarize_messages(
            self.messages, has_prior_summary=has_summary
        )
        self.messages = [
            {"role": "user", "content": SUMMARY_PREFIX + summary},
            {"role": "assistant", "content": "Understood. Continuing from the summary."},
        ]
        return summary

    async def _dispatch_tool(self, name: str, args: dict) -> str:
        if name in self.mcp_tool_names:
            return await call_mcp_tool(self.session, name, args)
        return execute_native_tool(name, args)

    async def chat(self, user_message: str) -> str:
        """Send one user message, run the tool-use loop, return the final reply."""
        self.messages.append({"role": "user", "content": user_message})
        folded = await self._trim_to_budget()
        if folded:
            print(f"  [trim] rolled {folded} oldest turn(s) into summary to stay "
                  f"under {self.max_input_tokens} tokens")

        for _ in range(self.max_turns):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                tools=self.tools,
                messages=self.messages,
            )
            self.last_input_tokens = response.usage.input_tokens
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

    def save(self, name: str) -> Path:
        SESSIONS_DIR.mkdir(exist_ok=True)
        path = SESSIONS_DIR / f"{name}.json"
        path.write_text(
            json.dumps(_serialize_messages(self.messages), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def load(self, name: str) -> int:
        path = SESSIONS_DIR / f"{name}.json"
        self.messages = json.loads(path.read_text(encoding="utf-8"))
        return len(self.messages)

    @staticmethod
    def list_sessions() -> list[str]:
        if not SESSIONS_DIR.exists():
            return []
        return sorted(p.stem for p in SESSIONS_DIR.glob("*.json"))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", help="Load a saved session by name on startup")
    parser.add_argument(
        "--max-input-tokens", type=int, default=100_000,
        help="Trigger auto-trim when input tokens exceed this (default 100000)",
    )
    parser.add_argument(
        "--keep-recent-turns", type=int, default=5,
        help="When auto-trimming, keep this many newest turns verbatim (default 5)",
    )
    cli_args = parser.parse_args()

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
            print("[hint] /save /load /list /tokens /compact /messages /reset /exit\n")

            agent = Agent(
                client=Anthropic(),
                session=session,
                tools=all_tools,
                mcp_tool_names=mcp_tool_names,
                max_input_tokens=cli_args.max_input_tokens,
                keep_recent_turns=cli_args.keep_recent_turns,
            )

            if cli_args.resume:
                try:
                    n = agent.load(cli_args.resume)
                    print(f"[resumed '{cli_args.resume}' with {n} messages]\n")
                except FileNotFoundError:
                    print(f"[no session '{cli_args.resume}', starting fresh]\n")

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
                if user_in == "/list":
                    names = Agent.list_sessions()
                    print(f"[sessions] {', '.join(names) if names else '(none)'}\n")
                    continue
                if user_in.startswith("/save"):
                    parts = user_in.split(maxsplit=1)
                    if len(parts) < 2:
                        print("[usage] /save <name>\n")
                        continue
                    path = agent.save(parts[1])
                    print(f"[saved {len(agent.messages)} messages to {path}]\n")
                    continue
                if user_in.startswith("/load"):
                    parts = user_in.split(maxsplit=1)
                    if len(parts) < 2:
                        print("[usage] /load <name>\n")
                        continue
                    try:
                        n = agent.load(parts[1])
                        print(f"[loaded '{parts[1]}' with {n} messages]\n")
                    except FileNotFoundError:
                        print(f"[no session '{parts[1]}']\n")
                    continue
                if user_in == "/tokens":
                    n = agent.count_tokens()
                    print(f"[tokens] {n} / {agent.max_input_tokens} "
                          f"({len(agent.messages)} messages)\n")
                    continue
                if user_in == "/messages":
                    agent.dump()
                    print()
                    continue
                if user_in == "/compact":
                    before = agent.count_tokens()
                    summary = await agent.compact()
                    after = agent.count_tokens()
                    print(f"[compacted {before} -> {after} tokens]")
                    print(f"[summary] {summary[:300]}{'...' if len(summary) > 300 else ''}\n")
                    continue

                reply = await agent.chat(user_in)
                print(f"\nclaude> {reply}  [tokens: {agent.last_input_tokens}]\n")


if __name__ == "__main__":
    asyncio.run(main())
