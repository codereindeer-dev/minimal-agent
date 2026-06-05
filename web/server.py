"""Minimal web UI for minimal-agent.

FastAPI + SSE + native HTML/JS. Boots one MCP session + one shared
WebAgent on startup, holds them on app.state for the lifetime of the
process. Lifecycle hooks fan agent events into per-request asyncio
queues; SSE endpoints drain those queues to the browser.

Endpoints
---------
POST /api/chat                  -> {"request_id"} (background agent.chat task)
GET  /api/stream?request_id=…   -> SSE: text / tool_start / tool_end /
                                        approval_request / assistant_done /
                                        usage / error / done
POST /api/approve               -> resolves a pending approval Future

GET  /api/state                 -> provider, model, tokens, message history
POST /api/reset | /api/compact
GET  /api/sessions   POST /api/sessions/{save,load}   DELETE /api/sessions/{name}
GET  /api/memories | /api/skills

GET  /api/providers             -> list available providers
POST /api/provider              -> hot-swap agent.provider + agent.model

Run:
    pip install fastapi uvicorn
    uvicorn web.server:app --reload
    open http://localhost:8000
"""

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel

from minimal_agent import (
    Agent,
    ANTHROPIC_MODEL,
    AnthropicProvider,
    DEFAULT_SYSTEM,
    LLMProvider,
    MemoryStore,
    NATIVE_TOOLS,
    OPENAI_MODEL,
    OpenAIProvider,
    PgVectorStore,
    SESSIONS_DIR,
    SkillsRegistry,
)

STATIC_DIR = Path(__file__).parent / "static"


def _messages_for_ui(messages: list) -> list[dict]:
    """Flatten Agent.messages into {role, text} entries for UI display.
    Skips tool_use / tool_result blocks — the chat view is conversational
    only; tool calls are an in-the-moment UI affordance, not history."""
    out = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            out.append({"role": role, "text": content})
            continue
        for block in content:
            btype = block.get("type")
            if btype == "text" and block.get("text"):
                out.append({"role": role, "text": block["text"]})
    return out


class WebAgent(Agent):
    """Agent variant whose run_shell approval flows through the active
    request's SSE queue + a Future resolved by POST /api/approve.

    Falls back to the base stdin prompt if no queue is currently bound,
    so unit-running this class outside the web server still works."""

    def __init__(self, *args, pending_approvals: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_approvals = pending_approvals
        self._current_queue: asyncio.Queue | None = None

    async def _approve_run_shell(self, command: str):
        if self.approval_mode == "auto":
            return True, None
        if self._current_queue is None:
            return await super()._approve_run_shell(command)
        approval_id = uuid.uuid4().hex
        fut = asyncio.get_event_loop().create_future()
        self._pending_approvals[approval_id] = fut
        self._current_queue.put_nowait({
            "type": "approval_request",
            "approval_id": approval_id,
            "command": command,
        })
        try:
            decision = await fut
        finally:
            self._pending_approvals.pop(approval_id, None)
        if decision == "approve":
            return True, None
        return False, "User declined via UI."


@asynccontextmanager
async def lifespan(app: FastAPI):
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

            # Memory backend is chosen at boot via the AGENT_MEMORY env var
            # ("jsonl" default, or "pgvector"). pgvector also needs DATABASE_URL.
            # There's no UI toggle on purpose: the two stores hold different
            # data, so swapping at runtime would silently change what the agent
            # can recall.
            backend = os.environ.get("AGENT_MEMORY", "jsonl").lower()
            try:
                if backend == "pgvector":
                    memory = PgVectorStore()
                    print(f"[web] memory: pgvector ({memory.count()} records)")
                else:
                    memory = MemoryStore()
                    print(f"[web] memory: jsonl ({memory.count()} records)")
            except Exception as e:
                memory = None
                print(f"[web] memory disabled: {e}")
            skills = SkillsRegistry()

            system = DEFAULT_SYSTEM
            if system is not None and skills.skills:
                system = system + "\n" + skills.system_addendum()

            provider = AnthropicProvider()
            app.state.pending_approvals = {}
            agent = WebAgent(
                pending_approvals=app.state.pending_approvals,
                provider=provider,
                session=session,
                tools=all_tools,
                mcp_tool_names=mcp_tool_names,
                memory=memory,
                skills=skills,
                system=system,
                approval_mode="ask",
                silent=True,
            )
            app.state.agent = agent
            app.state.lock = asyncio.Lock()
            app.state.requests = {}
            print(f"[web] agent ready (provider={provider.name}, "
                  f"model={agent.model}, tools={len(all_tools)})")
            print("[web] open http://localhost:8000")
            yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


class ChatRequest(BaseModel):
    message: str


def _truncate(s: str, n: int = 400) -> str:
    return s if len(s) <= n else s[:n] + f"\n... [{len(s) - n} more chars]"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    agent: WebAgent = app.state.agent
    lock: asyncio.Lock = app.state.lock
    requests: dict = app.state.requests

    request_id = uuid.uuid4().hex
    queue: asyncio.Queue = asyncio.Queue()
    requests[request_id] = queue

    # Track the most recently-fired pre_tool so post_tool can reuse the same
    # tool_call_id. _dispatch_tool runs sequentially, no interleaving.
    pending_tool_id: list[str] = []

    def on_chunk(chunk: str):
        queue.put_nowait({"type": "text", "chunk": chunk})

    def on_assistant(text: str):
        queue.put_nowait({"type": "assistant_done", "text": text})

    def on_post_turn(turn_idx, response, usage):
        queue.put_nowait({
            "type": "usage",
            "input": usage.input_tokens,
            "output": usage.output_tokens,
        })

    def on_pre_tool(name: str, args: dict):
        tool_call_id = uuid.uuid4().hex
        pending_tool_id.append(tool_call_id)
        queue.put_nowait({
            "type": "tool_start",
            "tool_call_id": tool_call_id,
            "name": name,
            "args": args,
        })

    def on_post_tool(name: str, args: dict, result: str, error):
        tool_call_id = pending_tool_id.pop() if pending_tool_id else None
        queue.put_nowait({
            "type": "tool_end",
            "tool_call_id": tool_call_id,
            "name": name,
            "result": _truncate(result),
            "blocked": error is not None,
        })

    hooks_registered = [
        ("text_chunk", on_chunk),
        ("assistant_message", on_assistant),
        ("post_turn", on_post_turn),
        ("pre_tool", on_pre_tool),
        ("post_tool", on_post_tool),
    ]

    async def run():
        for ev, fn in hooks_registered:
            agent.on(ev, fn)
        try:
            async with lock:
                agent._current_queue = queue
                try:
                    await agent.chat(req.message)
                finally:
                    agent._current_queue = None
        except Exception as e:
            queue.put_nowait({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            for ev, fn in hooks_registered:
                if fn in agent.hooks[ev]:
                    agent.hooks[ev].remove(fn)
            queue.put_nowait({"type": "done"})

    asyncio.create_task(run())
    return {"request_id": request_id}


@app.get("/api/stream")
async def stream(request_id: str):
    requests: dict = app.state.requests
    queue = requests.get(request_id)
    if queue is None:
        raise HTTPException(404, "unknown request_id")

    async def gen():
        try:
            while True:
                ev = await queue.get()
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                if ev["type"] == "done":
                    break
        finally:
            requests.pop(request_id, None)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class ApproveRequest(BaseModel):
    approval_id: str
    decision: str  # "approve" | "deny"


@app.post("/api/approve")
async def approve(req: ApproveRequest):
    pending: dict = app.state.pending_approvals
    fut = pending.get(req.approval_id)
    if fut is None:
        raise HTTPException(404, "unknown approval_id")
    if req.decision not in ("approve", "deny"):
        raise HTTPException(400, "decision must be 'approve' or 'deny'")
    if not fut.done():
        fut.set_result(req.decision)
    return {"ok": True}


# ─── Sidebar: sessions / memory / skills / status ────────────────────────────


@app.get("/api/state")
async def get_state():
    agent: WebAgent = app.state.agent
    return {
        "provider": agent.provider.name,
        "model": agent.model,
        "tokens": await agent.count_tokens(),
        "max_tokens": agent.max_input_tokens,
        "messages": _messages_for_ui(agent.messages),
        "n_messages": len(agent.messages),
    }


@app.post("/api/reset")
async def reset():
    agent: WebAgent = app.state.agent
    async with app.state.lock:
        agent.reset()
    return {"ok": True}


@app.post("/api/compact")
async def compact():
    agent: WebAgent = app.state.agent
    async with app.state.lock:
        before = await agent.count_tokens()
        summary = await agent.compact()
        after = await agent.count_tokens()
    return {"before": before, "after": after, "summary": summary}


@app.get("/api/sessions")
async def list_sessions():
    return {"sessions": Agent.list_sessions()}


class SessionRequest(BaseModel):
    name: str


@app.post("/api/sessions/save")
async def save_session(req: SessionRequest):
    agent: WebAgent = app.state.agent
    async with app.state.lock:
        path = agent.save(req.name)
    return {"path": str(path), "n_messages": len(agent.messages)}


@app.post("/api/sessions/load")
async def load_session(req: SessionRequest):
    agent: WebAgent = app.state.agent
    async with app.state.lock:
        try:
            n = agent.load(req.name)
        except FileNotFoundError:
            raise HTTPException(404, f"no session '{req.name}'")
    return {
        "n_messages": n,
        "messages": _messages_for_ui(agent.messages),
    }


@app.delete("/api/sessions/{name}")
async def delete_session(name: str):
    path = SESSIONS_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, f"no session '{name}'")
    path.unlink()
    return {"ok": True}


@app.get("/api/memories")
async def list_memories():
    agent: WebAgent = app.state.agent
    if agent.memory is None:
        return {"enabled": False, "backend": None, "count": 0, "memories": []}
    return {
        "enabled": True,
        "backend": type(agent.memory).__name__,
        "count": agent.memory.count(),
        "memories": agent.memory.all(),
    }


@app.get("/api/memories/search")
async def search_memories(q: str, top_k: int = 5):
    agent: WebAgent = app.state.agent
    if agent.memory is None:
        return {"enabled": False, "results": []}
    if not q.strip():
        return {"enabled": True, "results": []}
    async with app.state.lock:
        results = agent.memory.search(q, top_k=top_k)
    return {"enabled": True, "results": results}


@app.delete("/api/memories/{rec_id}")
async def delete_memory(rec_id: str):
    agent: WebAgent = app.state.agent
    if agent.memory is None:
        raise HTTPException(404, "memory disabled")
    async with app.state.lock:
        removed = agent.memory.delete(rec_id)
    if not removed:
        raise HTTPException(404, f"no memory '{rec_id}'")
    return {"ok": True, "count": agent.memory.count()}


@app.get("/api/skills")
async def list_skills():
    agent: WebAgent = app.state.agent
    if agent.skills is None:
        return {"skills": []}
    return {"skills": agent.skills.list()}


# ─── Provider / model switching ──────────────────────────────────────────────


@app.get("/api/providers")
async def list_providers():
    agent: WebAgent = app.state.agent
    return {
        "providers": [
            {"name": "anthropic", "default_model": ANTHROPIC_MODEL},
            {"name": "openai", "default_model": OPENAI_MODEL},
        ],
        "current": {"provider": agent.provider.name, "model": agent.model},
    }


class ProviderRequest(BaseModel):
    provider: str
    model: str | None = None


@app.post("/api/provider")
async def set_provider(req: ProviderRequest):
    agent: WebAgent = app.state.agent
    if req.provider == "openai":
        new_provider: LLMProvider = OpenAIProvider()
    elif req.provider == "anthropic":
        new_provider = AnthropicProvider()
    else:
        raise HTTPException(400, f"unknown provider '{req.provider}'")
    async with app.state.lock:
        agent.provider = new_provider
        agent.model = req.model or new_provider.default_model
    return {"provider": agent.provider.name, "model": agent.model}


