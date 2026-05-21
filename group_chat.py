"""Peer-to-peer multi-agent demo on top of minimal_agent.

A small GroupChat orchestrator holds N stateful specialist Agents and a
shared transcript. Each round a speaker-selection policy picks the next
agent; that agent receives the previous peer's reply (tagged with the
speaker name) as a user message, runs its tool-use loop to completion,
and the orchestrator broadcasts its reply to the others for the next
round. Terminates when an agent emits "[DONE]" or max_rounds is reached.

minimal_agent.py is NOT modified — Agent.chat(str) -> str is treated as
a black box. Each Agent keeps its own self.messages, so peers are truly
stateful (each remembers what it said last time it spoke).

Demo: planner + coder + reviewer collaborating on a small coding task.

    python group_chat.py "Write a Python fib(n) function with a test."
"""

import argparse
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from minimal_agent import (
    Agent,
    AnthropicProvider,
    LLMProvider,
    MemoryStore,
    NATIVE_TOOLS,
    OpenAIProvider,
    SkillsRegistry,
)


PLANNER_PROMPT = """\
You are PLANNER on a three-agent team (planner, coder, reviewer).

YOUR ONLY JOB: produce a short plan and hand off to coder. That is all.

STRICT RULES — DO NOT VIOLATE:
- DO NOT use any tools. No write_file, no run_shell, no read_file, nothing.
- DO NOT write code yourself. Coder writes code, not you.
- DO NOT review code yourself. Reviewer reviews, not you.
- DO NOT impersonate, simulate, or speak on behalf of coder or reviewer.
- DO NOT mark anything "Approved" — only the real reviewer can do that.

Your output for the first turn must be plain text only:
1. A numbered plan with 2-4 short steps.
2. The last line: "Handing off to coder."

Then wait. Coder will write code, reviewer will review, and the message
will come back to you. Only when reviewer has clearly approved the
final result do you reply with "[DONE]" on its own line — that ends
the conversation. Only you may end it.
"""

CODER_PROMPT = """\
You are CODER on a three-agent team (planner, coder, reviewer).
Implement what planner asked for. Use write_file to save code; use
run_shell to sanity-check it. Then hand back to reviewer with a 1-2
sentence summary of what you wrote. Do NOT say "[DONE]" — only planner ends.
"""

REVIEWER_PROMPT = """\
You are REVIEWER on a three-agent team (planner, coder, reviewer).
Read coder's output (use read_file if needed). Point out bugs, missing
edge cases, or style issues — be concise. If the code is good, say so
and hand back to planner. If not, hand back to coder with a numbered list.
Do NOT say "[DONE]" — only planner ends.
"""


class GroupChat:
    """Stateful peer agents share a transcript; orchestrator picks the next
    speaker each round and broadcasts their reply to the others."""

    DONE_SENTINEL = "[DONE]"

    def __init__(
        self,
        agents: dict[str, Agent],
        max_rounds: int = 20,
        select_next=None,
    ):
        if len(agents) < 2:
            raise ValueError("GroupChat needs at least 2 agents")
        self.agents = agents
        self.order = list(agents.keys())
        self.max_rounds = max_rounds
        self.select_next = select_next or self._round_robin
        self.transcript: list[dict] = []

    def _round_robin(self, transcript, last_speaker):
        if last_speaker is None:
            return self.order[0]
        i = self.order.index(last_speaker)
        return self.order[(i + 1) % len(self.order)]

    async def run(self, task: str) -> list[dict]:
        last_speaker = None
        next_msg = task
        # Sliding window size: under round-robin this equals
        # "messages said since this agent last spoke" — each peer sees
        # exactly what they missed, no more, no less.
        window_size = len(self.agents) - 1

        for r in range(self.max_rounds):
            speaker = self.select_next(self.transcript, last_speaker)
            agent = self.agents[speaker]

            print(f"\n─── round {r + 1}: {speaker} ───")
            reply = await agent.chat(next_msg)
            print((reply.strip() or "(no text)"))

            self.transcript.append({"speaker": speaker, "content": reply})

            if self.DONE_SENTINEL in reply:
                print(f"\n─── {speaker} ended the conversation ───")
                return self.transcript

            window = self.transcript[-window_size:]
            next_msg = "\n\n".join(
                f"[from {entry['speaker']}]\n{entry['content']}"
                for entry in window
            )
            last_speaker = speaker

        print(f"\n─── max_rounds={self.max_rounds} reached ───")
        return self.transcript


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        nargs="?",
        default="Write a Python function fib(n) that returns the nth Fibonacci number, plus a quick test that prints fib(10).",
    )
    parser.add_argument("--provider", choices=("anthropic", "openai"), default="anthropic")
    parser.add_argument("--max-rounds", type=int, default=20)
    cli = parser.parse_args()

    provider: LLMProvider = (
        OpenAIProvider() if cli.provider == "openai" else AnthropicProvider()
    )

    server_params = StdioServerParameters(command="python", args=["-m", "mcp_server_fetch"])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listing = await session.list_tools()
            mcp_tools = [
                {"name": t.name, "description": t.description or "", "input_schema": t.inputSchema}
                for t in listing.tools
            ]
            mcp_tool_names = {t["name"] for t in mcp_tools}
            all_tools = NATIVE_TOOLS + mcp_tools

            try:
                memory = MemoryStore()
            except Exception:
                memory = None
            skills = SkillsRegistry()

            # Peers are inside a multi-agent layer (group chat). Stripping
            # `spawn_agent` prevents a peer from opening another multi-agent
            # layer underneath itself and bypassing the group entirely.
            # Stripping `remember` keeps long-term memory writes for the
            # user-facing layer only (same read/write asymmetry as CH09).
            PEER_FORBIDDEN = {"spawn_agent", "remember"}
            peer_tools = [t for t in all_tools if t["name"] not in PEER_FORBIDDEN]

            def make_agent(system: str) -> Agent:
                return Agent(
                    provider=provider,
                    session=session,
                    tools=peer_tools,
                    mcp_tool_names=mcp_tool_names,
                    memory=memory,
                    skills=skills,
                    system=system,
                    approval_mode="auto",
                    silent=True,
                    max_turns=10,
                )

            agents = {
                "planner": make_agent(PLANNER_PROMPT),
                "coder": make_agent(CODER_PROMPT),
                "reviewer": make_agent(REVIEWER_PROMPT),
            }

            print(f"[group] provider: {provider.name}")
            print(f"[group] agents:   {list(agents.keys())} (round-robin)")
            print(f"[group] task:     {cli.task}")

            chat = GroupChat(agents, max_rounds=cli.max_rounds)
            await chat.run(cli.task)


if __name__ == "__main__":
    asyncio.run(main())
