"""Minimal web UI for minimal-agent.

FastAPI + native HTML/JS. Boots one MCP session + one shared Agent on
startup, holds them on app.state for the request handlers. Same wiring
as minimal_agent.main() — provider, tools, memory, skills.

Endpoints
---------
POST /api/chat  -> {"reply", "tokens"}  (synchronous, blocks on agent.chat)

Run:
    pip install fastapi uvicorn
    uvicorn web.server:app --reload
    open http://localhost:8000
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
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
            )
            app.state.agent = agent
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
    reply = await agent.chat(req.message)
    return {"reply": reply, "tokens": agent.last_input_tokens}
