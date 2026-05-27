"""Minimal web UI for minimal-agent.

FastAPI + SSE + native HTML/JS. Boots one MCP session + one shared
Agent on startup, holds them on app.state for the lifetime of the
process. Lifecycle hooks fan agent events into per-request asyncio
queues; SSE endpoints drain those queues to the browser.

Endpoints
---------
POST /api/chat                  -> {"request_id"} (background agent.chat task)
GET  /api/stream?request_id=…   -> SSE: text / assistant_done /
                                        usage / error / done

Run:
    pip install fastapi uvicorn
    uvicorn web.server:app --reload
    open http://localhost:8000
"""

import asyncio
import json
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
    AnthropicProvider,
    DEFAULT_SYSTEM,
    MemoryStore,
    NATIVE_TOOLS,
    SkillsRegistry,
)

STATIC_DIR = Path(__file__).parent / "static"


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

            try:
                memory = MemoryStore()
            except Exception:
                memory = None
            skills = SkillsRegistry()

            system = DEFAULT_SYSTEM
            if system is not None and skills.skills:
                system = system + "\n" + skills.system_addendum()

            provider = AnthropicProvider()
            agent = Agent(
                provider=provider,
                session=session,
                tools=all_tools,
                mcp_tool_names=mcp_tool_names,
                memory=memory,
                skills=skills,
                system=system,
                approval_mode="auto",
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


@app.post("/api/chat")
async def chat(req: ChatRequest):
    agent: Agent = app.state.agent
    lock: asyncio.Lock = app.state.lock
    requests: dict = app.state.requests

    request_id = uuid.uuid4().hex
    queue: asyncio.Queue = asyncio.Queue()
    requests[request_id] = queue

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

    async def run():
        agent.on("text_chunk", on_chunk)
        agent.on("assistant_message", on_assistant)
        agent.on("post_turn", on_post_turn)
        try:
            async with lock:
                await agent.chat(req.message)
        except Exception as e:
            queue.put_nowait({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            for ev, fn in [
                ("text_chunk", on_chunk),
                ("assistant_message", on_assistant),
                ("post_turn", on_post_turn),
            ]:
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
