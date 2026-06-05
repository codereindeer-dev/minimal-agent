"""
Minimal agent loop — pluggable LLM provider (Anthropic or OpenAI) + native
tools + MCP server + RAG memory.

Run:
    pip install anthropic python-dotenv mcp mcp-server-fetch voyageai
    # Optional, for the OpenAI provider:
    #   pip install openai tiktoken
    # Optional, for the pgvector memory backend:
    #   pip install "psycopg[binary]"   (and a Postgres with the `vector` ext)
    # put ANTHROPIC_API_KEY and VOYAGE_API_KEY in .env
    # (and OPENAI_API_KEY if launching with `--provider openai`)
    # (and DATABASE_URL if launching with `--memory pgvector`)
    python minimal_agent.py                    # default: Anthropic + jsonl memory
    python minimal_agent.py --provider openai  # use OpenAI instead
    python minimal_agent.py --memory pgvector  # use Postgres/pgvector memory
"""

import argparse
import asyncio
import json
import math
import os
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import voyageai
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()
ANTHROPIC_MODEL = "claude-sonnet-4-6"
OPENAI_MODEL = "gpt-5.5"
MODEL = ANTHROPIC_MODEL  # back-compat alias for callers that import MODEL
VOYAGE_MODEL = "voyage-3-lite"  # 512-dim, cheap, fast
VOYAGE_DIM = 512  # embedding dimensionality of VOYAGE_MODEL (must match the
                  # pgvector column type vector(VOYAGE_DIM))
MEMORY_BACKENDS = ("jsonl", "pgvector")
SESSIONS_DIR = Path("sessions")
MEMORY_DIR = Path("memory")
MEMORY_FILE = MEMORY_DIR / "store.jsonl"
SKILLS_DIR = Path("skills")
SUMMARY_PREFIX = "[Earlier conversation summary]\n"

APPROVAL_MODES = ("auto", "ask", "safe")
SAFE_SHELL_PREFIXES = (
    "ls", "pwd", "cat", "head", "tail", "wc", "file", "which", "whereis",
    "grep", "find", "tree", "echo",
    "git status", "git log", "git diff", "git show", "git branch", "git remote",
)
SHELL_CHAIN_CHARS = ("|", ">", "<", "&", ";", "`", "$")


def is_safe_shell(command: str) -> bool:
    """True if `command` matches the read-only allow-list AND uses no shell
    chaining metacharacters that could escalate to destructive ops."""
    if any(ch in command for ch in SHELL_CHAIN_CHARS):
        return False
    cmd = command.strip().lower()
    return any(
        cmd == p or cmd.startswith(p + " ") for p in SAFE_SHELL_PREFIXES
    )


DEFAULT_SYSTEM = """\
You are a coding agent running in a terminal REPL on the user's machine.

Memory tools:
- `recall` searches long-term memory across sessions. Call it BEFORE
  answering questions about the user's preferences, history, prior
  decisions, project context, or anything they might have told you in
  a previous session. Do not assume the current conversation contains
  everything you know.
- `remember` saves a single fact, preference, or decision worth keeping
  across sessions. Call it when the user shares such information —
  their name, preferences, project context, conclusions reached.

Sub-agents:
- For research-heavy tasks (searching across many files, multi-step
  exploration whose intermediate results you don't need), prefer
  `spawn_agent` over doing it yourself. The sub-agent has its own
  context, so its grep/read noise won't bloat your history.
- Don't spawn for simple lookups — the sub-agent has cold context and
  takes longer to ramp up than just running one tool yourself.

Style:
- Be concise. Skip preambles like "I will now..." and never restate
  the user's question. Lead with the action or the answer.
- Use backticks for file paths, commands, and identifiers.
- Reply in the same language the user is using.

Safety:
- Before calling `run_shell`, say in one short sentence what the
  command does and why. The user is reading along and may stop you.
- Prefer read-only commands (`ls`, `cat`, `grep`, `git status`, etc.)
  when gathering information.
- Never run destructive commands (`rm -rf`, `git reset --hard`,
  `git push --force`, etc.) unless the user has explicitly asked for
  that exact operation.
"""

MAX_DEPTH = 2  # depth 0 = top-level user agent; depth 1 = sub-agent; no further

SUBAGENT_SYSTEM = """\
You are a focused sub-agent spawned to complete one specific task.

You have NO access to the parent agent's conversation history. The task
you were given is self-contained — work from what's in front of you.

Style:
- Do the task. Don't chat. Don't ask for clarification unless truly
  blocked.
- When done, respond with a focused, factual answer. No preamble like
  "Here is what I found:" — just the result.
- Reply in the same language the task is using.

Constraints:
- You cannot spawn further sub-agents.
- You can read long-term memory (`recall`) but not write to it
  (`remember` is unavailable).
- Tool gating (e.g. `run_shell` approval) inherits from the parent
  session and still applies.
"""

# Lifecycle events fired by Agent.chat() / _dispatch_tool().
# Callback signatures:
#   "user_message"      (text: str)
#   "pre_turn"          (turn_idx: int, messages: list)
#   "text_chunk"        (chunk: str)        fires for each streamed text delta
#   "post_turn"         (turn_idx: int, response, usage)
#   "pre_tool"          (name: str, args: dict)   raise HookBlocked to veto
#   "post_tool"         (name: str, args: dict, result: str,
#                        error: HookBlocked | None)
#   "assistant_message" (text: str)
HOOK_EVENTS = (
    "user_message",
    "pre_turn",
    "text_chunk",
    "post_turn",
    "pre_tool",
    "post_tool",
    "assistant_message",
)


class HookBlocked(Exception):
    """Raised by a `pre_tool` hook to veto a tool call. The `result` string
    is fed back to the model in place of real execution, so the model sees
    a normal tool_result and can react (retry differently, give up, etc.)."""

    def __init__(self, result: str = "blocked by hook"):
        super().__init__(result)
        self.result = result


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
    {
        "name": "remember",
        "description": (
            "Save a fact, preference, or note to long-term memory that persists across "
            "sessions. Use this when the user shares something worth remembering — "
            "their name, preferences, project context, decisions, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The information to remember"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional category tags",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "recall",
        "description": (
            "Search long-term memory via semantic similarity. Use this whenever the "
            "user might be referencing something stored from a previous session, or "
            "before answering questions about their preferences/history/context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
                "top_k": {
                    "type": "integer",
                    "description": "How many results to return (default 3)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "load_skill",
        "description": (
            "Load the full instructions for a named skill. Skills are pre-written "
            "task guides (e.g. how to work with PDFs, how to write a particular "
            "kind of code) provided by the user. The system prompt lists the "
            "available skill names with one-line descriptions; call this tool to "
            "fetch the actual instructions when one of those skills matches the "
            "current task. The returned text may also list supporting files in "
            "the skill's directory that you can read via `read_file` on demand."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name as listed in the system prompt",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "spawn_agent",
        "description": (
            "Spawn a focused sub-agent to handle one specific task. The sub-agent "
            "has its OWN context — it CANNOT see your conversation history, so the "
            "task description must be fully self-contained. Use this for: "
            "(1) searching/exploring across many files without polluting your "
            "context, (2) getting an independent second opinion, (3) multi-step "
            "research whose intermediate steps you don't need to see. Returns the "
            "sub-agent's final answer as a string. The sub-agent cannot itself "
            "spawn further sub-agents and cannot write to long-term memory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "Self-contained task description. Include any file paths, "
                        "constraints, or context the sub-agent needs — it sees "
                        "nothing else from this conversation."
                    ),
                },
                "allowed_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional subset of tool names to expose to the sub-agent. "
                        "Defaults to all tools except `spawn_agent` and `remember`."
                    ),
                },
            },
            "required": ["task"],
        },
    },
]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class MemoryStore:
    """Append-only JSONL store of {id, text, tags, embedding, created} records.
    Embeddings via Voyage AI; search via cosine similarity in memory."""

    def __init__(self, path: Path = MEMORY_FILE, model: str = VOYAGE_MODEL):
        self.path = path
        self.model = model
        self.client = voyageai.Client()  # reads VOYAGE_API_KEY from env
        self.records: list[dict] = []
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        # input_type: "document" when storing, "query" when searching
        result = self.client.embed(texts, model=self.model, input_type=input_type)
        return result.embeddings

    def add(self, text: str, tags: list[str] | None = None) -> str:
        embedding = self._embed([text], "document")[0]
        record = {
            "id": uuid.uuid4().hex[:8],
            "text": text,
            "tags": tags or [],
            "embedding": embedding,
            "created": time.time(),
        }
        self.records.append(record)
        MEMORY_DIR.mkdir(exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record["id"]

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not self.records:
            return []
        q_emb = self._embed([query], "query")[0]
        scored = [(_cosine(q_emb, r["embedding"]), r) for r in self.records]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "tags": r["tags"],
                "score": round(s, 3),
            }
            for s, r in scored[:top_k]
        ]

    def all(self) -> list[dict]:
        return [
            {"id": r["id"], "text": r["text"], "tags": r["tags"]}
            for r in self.records
        ]

    def count(self) -> int:
        return len(self.records)

    def delete(self, rec_id: str) -> bool:
        """Remove a record by id. JSONL is append-only, so we drop it from the
        in-memory list and rewrite the whole file. Returns True if a record was
        actually removed."""
        before = len(self.records)
        self.records = [r for r in self.records if r["id"] != rec_id]
        if len(self.records) == before:
            return False
        MEMORY_DIR.mkdir(exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return True


def _vec_literal(embedding: list[float]) -> str:
    """Render an embedding as pgvector's text input format: '[1.0,2.0,...]'.

    We send the vector as a text literal cast with `::vector` in SQL rather than
    relying on the optional `pgvector` Python adapter. This keeps the dependency
    surface to just `psycopg` + the server-side `vector` extension, and sidesteps
    psycopg's ambiguity between a plain float list (→ float8[]) and a vector."""
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


class PgVectorStore:
    """pgvector-backed long-term memory. Drop-in for MemoryStore: exposes the
    same add() / search() / all() interface, so the Agent and the
    remember/recall tools can't tell which backend is underneath.

    Storage + ANN search live in Postgres (one `memory` table with an HNSW
    index on a vector(VOYAGE_DIM) column). Embeddings still come from Voyage AI;
    similarity is the cosine-distance operator `<=>`, so ORDER BY ... LIMIT k
    is a real indexed nearest-neighbour query instead of MemoryStore's O(n)
    in-memory scan.

    Connection string comes from the `dsn` arg or the DATABASE_URL env var, e.g.
        postgresql://user:pass@localhost:5432/agent
    Requires `pip install "psycopg[binary]"` and the `vector` extension
    available on the server (CREATE EXTENSION is attempted on connect)."""

    def __init__(
        self,
        dsn: str | None = None,
        model: str = VOYAGE_MODEL,
        dim: int = VOYAGE_DIM,
        table: str = "memory",
    ):
        try:
            import psycopg
        except ImportError as e:
            raise ImportError(
                'pgvector backend requires `pip install "psycopg[binary]"`'
            ) from e
        dsn = dsn or os.environ.get("DATABASE_URL")
        if not dsn:
            raise ValueError(
                "no Postgres DSN: pass dsn= or set DATABASE_URL "
                "(e.g. postgresql://user:pass@localhost:5432/agent)"
            )
        self.model = model
        self.dim = dim
        self.table = table
        self.dsn = dsn
        self._psycopg = psycopg
        self.client = voyageai.Client()  # reads VOYAGE_API_KEY from env
        self._connect()
        self._ensure_schema()

    def _connect(self):
        # autocommit so each add() persists immediately, mirroring the JSONL
        # store's append-on-write semantics.
        self.conn = self._psycopg.connect(self.dsn, autocommit=True)

    def _execute(self, sql: str, params=None):
        """Run a statement, transparently reconnecting once if the connection
        went stale. Long-running web servers keep this store alive for hours;
        serverless Postgres (e.g. Neon) drops idle connections, so the first
        query after a lull would otherwise fail with an OperationalError."""
        try:
            return self.conn.execute(sql, params)
        except self._psycopg.OperationalError:
            # Connection died (idle timeout, server suspend, SSL drop). Rebuild
            # it and retry once; a second failure is a real error, so let it raise.
            try:
                self.conn.close()
            except Exception:
                pass
            self._connect()
            return self.conn.execute(sql, params)

    def _ensure_schema(self):
        try:
            self._execute("CREATE EXTENSION IF NOT EXISTS vector")
        except Exception as e:
            raise RuntimeError(
                "could not enable the `vector` extension (needs a superuser or "
                "a role with CREATE privilege, or have a DBA run "
                "`CREATE EXTENSION vector;` once): " + str(e)
            ) from e
        self._execute(
            f"CREATE TABLE IF NOT EXISTS {self.table} ("
            "  id        TEXT PRIMARY KEY,"
            "  text      TEXT NOT NULL,"
            "  tags      TEXT[] NOT NULL DEFAULT '{}',"
            f"  embedding vector({self.dim}) NOT NULL,"
            "  created   DOUBLE PRECISION NOT NULL"
            ")"
        )
        # HNSW index for cosine distance. IF NOT EXISTS keeps startup idempotent.
        self._execute(
            f"CREATE INDEX IF NOT EXISTS {self.table}_embedding_idx "
            f"ON {self.table} USING hnsw (embedding vector_cosine_ops)"
        )

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        result = self.client.embed(texts, model=self.model, input_type=input_type)
        return result.embeddings

    def add(self, text: str, tags: list[str] | None = None) -> str:
        rec_id = uuid.uuid4().hex[:8]
        embedding = self._embed([text], "document")[0]
        self._execute(
            f"INSERT INTO {self.table} (id, text, tags, embedding, created) "
            "VALUES (%s, %s, %s, %s::vector, %s)",
            (rec_id, text, tags or [], _vec_literal(embedding), time.time()),
        )
        return rec_id

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        q_emb = self._embed([query], "query")[0]
        q_lit = _vec_literal(q_emb)
        # `<=>` is cosine distance (0 = identical). Convert to a similarity
        # score in [0, 1] so the shape matches MemoryStore.search().
        cur = self._execute(
            f"SELECT id, text, tags, 1 - (embedding <=> %s::vector) AS score "
            f"FROM {self.table} "
            f"ORDER BY embedding <=> %s::vector "
            "LIMIT %s",
            (q_lit, q_lit, top_k),
        )
        return [
            {"id": r[0], "text": r[1], "tags": r[2], "score": round(float(r[3]), 3)}
            for r in cur.fetchall()
        ]

    def all(self) -> list[dict]:
        cur = self._execute(
            f"SELECT id, text, tags FROM {self.table} ORDER BY created"
        )
        return [{"id": r[0], "text": r[1], "tags": r[2]} for r in cur.fetchall()]

    def count(self) -> int:
        cur = self._execute(f"SELECT count(*) FROM {self.table}")
        return cur.fetchone()[0]

    def delete(self, rec_id: str) -> bool:
        cur = self._execute(
            f"DELETE FROM {self.table} WHERE id = %s", (rec_id,)
        )
        return cur.rowcount > 0


def _parse_skill_frontmatter(text: str) -> tuple[dict, str]:
    """Parse minimal YAML frontmatter (--- ... ---) from a SKILL.md file.
    Supports only top-level `key: value` lines — no nesting, no lists.
    Returns (frontmatter_dict, body_text)."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    fm_block = text[3:end].strip()
    body = text[end + 4:].lstrip("\n")
    fm: dict = {}
    for line in fm_block.splitlines():
        line = line.rstrip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        fm[k.strip()] = v.strip().strip('"').strip("'")
    return fm, body


class SkillsRegistry:
    """Discovers SKILL.md files under a skills root and loads them on demand.

    Each skill is a folder with SKILL.md at its root:
        skills/
          git-helper/
            SKILL.md       # frontmatter: name, description + body instructions
            scripts/...    # optional supporting files (model reads via read_file)

    Only the {name, description} pairs go into the system prompt (cheap).
    The body is loaded only when the model calls `load_skill` for that name."""

    def __init__(self, root: Path = SKILLS_DIR):
        self.root = root
        self.skills: dict[str, dict] = {}
        self._discover()

    def _discover(self):
        if not self.root.exists():
            return
        for skill_md in sorted(self.root.glob("*/SKILL.md")):
            try:
                text = skill_md.read_text(encoding="utf-8")
            except Exception:
                continue
            fm, body = _parse_skill_frontmatter(text)
            name = fm.get("name") or skill_md.parent.name
            self.skills[name] = {
                "name": name,
                "description": fm.get("description", ""),
                "dir": skill_md.parent,
                "body": body,
            }

    def list(self) -> list[dict]:
        return [
            {"name": s["name"], "description": s["description"]}
            for s in self.skills.values()
        ]

    def load(self, name: str) -> str:
        if name not in self.skills:
            available = ", ".join(self.skills.keys()) or "(none)"
            return f"ERROR: unknown skill '{name}'. Available: {available}"
        s = self.skills[name]
        # Surface supporting files so the model knows what's available without
        # us pulling their content into context up front (progressive disclosure).
        extras = sorted(
            (s["dir"] / p).as_posix()
            for p in (q.relative_to(s["dir"]) for q in s["dir"].rglob("*"))
            if (s["dir"] / p).is_file() and p.name != "SKILL.md"
        )
        out = [f"# Skill: {s['name']}", "", s["body"].rstrip()]
        if extras:
            out += ["", "Additional files in this skill (read with `read_file` as needed):"]
            out += [f"  - {p}" for p in extras]
        return "\n".join(out)

    def system_addendum(self) -> str:
        """Short blurb to append to the system prompt: names + one-liners only.
        Empty string when no skills are present, so caller can blindly concat."""
        if not self.skills:
            return ""
        lines = ["", "Available skills (call `load_skill` with the name to load full instructions):"]
        for s in self.skills.values():
            lines.append(f"- {s['name']}: {s['description']}")
        return "\n".join(lines)


def execute_native_tool(
    name: str,
    args: dict,
    memory: MemoryStore | None = None,
    skills: SkillsRegistry | None = None,
) -> str:
    if name == "run_shell":
        proc = subprocess.run(
            args["command"], shell=True, capture_output=True, text=True, timeout=30
        )
        return json.dumps({
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }, ensure_ascii=False)
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
    if name == "remember":
        if memory is None:
            return "ERROR: memory store not initialized"
        try:
            rec_id = memory.add(args["text"], args.get("tags"))
            return f"Remembered as id={rec_id}"
        except Exception as e:
            return f"ERROR remembering: {e}"
    if name == "recall":
        if memory is None:
            return "ERROR: memory store not initialized"
        try:
            results = memory.search(args["query"], args.get("top_k", 3))
            return json.dumps(results, ensure_ascii=False)
        except Exception as e:
            return f"ERROR recalling: {e}"
    if name == "load_skill":
        if skills is None:
            return "ERROR: skills registry not initialized"
        return skills.load(args["name"])
    return f"ERROR: unknown native tool {name}"


def _serialize_messages(messages: list) -> list:
    """Identity pass-through. Messages are stored in the canonical dict form
    already (NormalizedResponse returns dict blocks), so save/load is JSON-safe
    without any conversion. Kept as a seam in case a future provider returns
    objects that need flattening."""
    return messages


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


# ---------------------------------------------------------------------------
# LLM provider abstraction
#
# The Agent only talks to its model through `LLMProvider`. Swapping Anthropic
# for OpenAI (or anything else) is a one-line change in main(). The canonical
# message/tool format the Agent stores internally is Anthropic-shaped — the
# AnthropicProvider passes it through unchanged, the OpenAIProvider translates
# on the way out (and normalizes the response on the way back in).
# ---------------------------------------------------------------------------


@dataclass
class NormalizedUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class NormalizedResponse:
    """Provider-agnostic response. `content` is a list of dict blocks:
        {"type": "text", "text": "..."}
        {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
    `stop_reason` is normalized to "end_turn" or "tool_use"."""
    content: list[dict]
    stop_reason: str
    usage: NormalizedUsage


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    name: str
    default_model: str

    @abstractmethod
    def format_tools(self, tools: list) -> list:
        """Convert canonical (Anthropic-shaped) tool defs to provider format."""

    @abstractmethod
    def stream(self, *, messages: list, tools: list, system: str | None,
               model: str, max_tokens: int):
        """Return an async context manager exposing `.text_stream`
        (AsyncIterator[str]) and `async get_final_message()` ->
        NormalizedResponse."""

    @abstractmethod
    async def create(self, *, messages: list, system: str | None,
                     model: str, max_tokens: int) -> NormalizedResponse:
        """Non-streaming completion (used for history summaries)."""

    @abstractmethod
    async def count_tokens(self, *, messages: list, tools: list,
                           system: str | None, model: str) -> int:
        """Estimate input tokens for the current state."""


def _anthropic_to_normalized(resp) -> NormalizedResponse:
    content = []
    for block in resp.content:
        if block.type == "text":
            content.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            content.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
    return NormalizedResponse(
        content=content,
        stop_reason=resp.stop_reason or "end_turn",
        usage=NormalizedUsage(
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        ),
    )


class _AnthropicStream:
    """Wraps the Anthropic SDK async stream so its final message is returned
    in NormalizedResponse shape — the agent never sees SDK-specific blocks.
    Async context manager + async iterator throughout."""

    def __init__(self, client: AsyncAnthropic, kwargs: dict):
        self.client = client
        self.kwargs = kwargs

    async def __aenter__(self):
        self._cm = self.client.messages.stream(**self.kwargs)
        self._inner = await self._cm.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return await self._cm.__aexit__(exc_type, exc, tb)

    @property
    def text_stream(self):
        return self._inner.text_stream         # AsyncIterator[str]

    async def get_final_message(self) -> NormalizedResponse:
        return _anthropic_to_normalized(await self._inner.get_final_message())


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    default_model = ANTHROPIC_MODEL

    def __init__(self):
        # AsyncAnthropic so stream / create / count_tokens are non-blocking
        # — the underlying SDK call awaits the HTTP read instead of holding
        # the event loop, which matters once multiple async tasks (e.g. a
        # web SSE generator) share the loop with agent.chat().
        self.client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env

    def format_tools(self, tools):
        return tools  # canonical format IS Anthropic's

    def stream(self, *, messages, tools, system, model, max_tokens):
        kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if system:
            kwargs["system"] = system
        return _AnthropicStream(self.client, kwargs)

    async def create(self, *, messages, system, model, max_tokens):
        kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        return _anthropic_to_normalized(await self.client.messages.create(**kwargs))

    async def count_tokens(self, *, messages, tools, system, model):
        if not messages:
            return 0
        kwargs = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if system:
            kwargs["system"] = system
        result = await self.client.messages.count_tokens(**kwargs)
        return result.input_tokens


class _OpenAIStream:
    """Wraps an OpenAI Responses API stream so it exposes the same
    `text_stream` / `get_final_message()` shape used for Anthropic. The
    Responses API emits typed SSE events (output_item.added, output_text.delta,
    function_call_arguments.delta, completed, …); we accumulate text + tool
    calls per output-item and emit a NormalizedResponse at the end."""

    def __init__(self, client, kwargs: dict):
        self.client = client
        self.kwargs = kwargs
        self._final: NormalizedResponse | None = None

    async def __aenter__(self):
        self._stream = await self.client.responses.create(**self.kwargs)
        self._iter = self._iter_events()
        return self

    async def __aexit__(self, *args):
        return False

    @property
    def text_stream(self):
        return self._iter

    async def _iter_events(self):
        # Accumulate per output-item state, keyed by item_id, preserving
        # encounter order so interleaved text / tool_use blocks come out
        # in the same sequence the model emitted them.
        items: dict[str, dict] = {}
        order: list[str] = []
        usage = None

        def ensure_item(item_id: str, kind: str, **defaults) -> dict:
            if item_id not in items:
                entry = {"kind": kind, **defaults}
                items[item_id] = entry
                order.append(item_id)
            return items[item_id]

        async for event in self._stream:
            etype = getattr(event, "type", None)

            if etype == "response.output_item.added":
                item = getattr(event, "item", None)
                if item is None:
                    continue
                item_type = getattr(item, "type", None)
                item_id = getattr(item, "id", None)
                if not item_id:
                    continue
                if item_type == "message":
                    ensure_item(item_id, "text", parts=[])
                elif item_type == "function_call":
                    ensure_item(
                        item_id, "tool_use",
                        call_id=getattr(item, "call_id", "") or "",
                        name=getattr(item, "name", "") or "",
                        arguments="",
                    )
                # reasoning items: ignored — we don't surface chain-of-thought

            elif etype == "response.output_text.delta":
                item_id = getattr(event, "item_id", None)
                delta = getattr(event, "delta", "") or ""
                if item_id:
                    entry = ensure_item(item_id, "text", parts=[])
                    entry["parts"].append(delta)
                if delta:
                    yield delta

            elif etype == "response.function_call_arguments.delta":
                item_id = getattr(event, "item_id", None)
                delta = getattr(event, "delta", "") or ""
                if item_id:
                    entry = ensure_item(
                        item_id, "tool_use",
                        call_id="", name="", arguments="",
                    )
                    entry["arguments"] += delta

            elif etype == "response.completed":
                resp = getattr(event, "response", None)
                if resp is not None:
                    usage = getattr(resp, "usage", None)
                    # Backfill call_id / name from final output items in case
                    # output_item.added arrived without them (defensive).
                    for it in (getattr(resp, "output", []) or []):
                        if getattr(it, "type", None) != "function_call":
                            continue
                        it_id = getattr(it, "id", None)
                        if it_id and it_id in items:
                            entry = items[it_id]
                            if not entry.get("call_id"):
                                entry["call_id"] = getattr(it, "call_id", "") or ""
                            if not entry.get("name"):
                                entry["name"] = getattr(it, "name", "") or ""

            elif etype in ("response.failed", "error"):
                err = getattr(event, "error", None) or event
                raise RuntimeError(f"OpenAI Responses stream error ({etype}): {err}")

        content = []
        for item_id in order:
            entry = items[item_id]
            if entry["kind"] == "text":
                text = "".join(entry.get("parts", []))
                if text:
                    content.append({"type": "text", "text": text})
            else:  # tool_use
                raw = entry.get("arguments") or ""
                try:
                    inp = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    inp = {}
                content.append({
                    "type": "tool_use",
                    "id": entry.get("call_id") or "",
                    "name": entry.get("name") or "",
                    "input": inp,
                })

        has_tool = any(b["type"] == "tool_use" for b in content)
        in_tok = getattr(usage, "input_tokens", 0) if usage else 0
        out_tok = getattr(usage, "output_tokens", 0) if usage else 0
        self._final = NormalizedResponse(
            content=content,
            stop_reason="tool_use" if has_tool else "end_turn",
            usage=NormalizedUsage(input_tokens=in_tok, output_tokens=out_tok),
        )

    async def get_final_message(self) -> NormalizedResponse:
        if self._final is None:
            # Drain remaining events (covers tool-only responses where the
            # caller never iterated text_stream because nothing streamed).
            async for _ in self._iter:
                pass
        assert self._final is not None
        return self._final


def _responses_to_normalized(resp) -> NormalizedResponse:
    """Convert a non-streaming Responses API result into our canonical shape.
    Walk the `output` array in order so text + tool_use blocks keep the model's
    emission sequence; ignore reasoning items (we never surface them)."""
    content = []
    for item in (getattr(resp, "output", []) or []):
        item_type = getattr(item, "type", None)
        if item_type == "message":
            for part in (getattr(item, "content", []) or []):
                if getattr(part, "type", None) == "output_text":
                    content.append({
                        "type": "text",
                        "text": getattr(part, "text", "") or "",
                    })
        elif item_type == "function_call":
            raw = getattr(item, "arguments", "") or ""
            try:
                inp = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                inp = {}
            content.append({
                "type": "tool_use",
                "id": getattr(item, "call_id", "") or "",
                "name": getattr(item, "name", "") or "",
                "input": inp,
            })
        # reasoning items: skipped
    has_tool = any(b["type"] == "tool_use" for b in content)
    usage = getattr(resp, "usage", None)
    return NormalizedResponse(
        content=content,
        stop_reason="tool_use" if has_tool else "end_turn",
        usage=NormalizedUsage(
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
        ),
    )


class OpenAIProvider(LLMProvider):
    """Talks to OpenAI via /v1/responses (NOT /v1/chat/completions).

    The Responses API is OpenAI's recommended path for reasoning models +
    tools — chat.completions rejects `tools + reasoning_effort` on gpt-5.x.
    Responses also unifies the budget parameter (`max_output_tokens` for
    everything) and uses a cleaner input/output item model, so we avoid the
    legacy two-parameter, two-message-shape split entirely.
    """

    name = "openai"
    default_model = OPENAI_MODEL

    # Models that accept the `reasoning` parameter. For an interactive agent
    # loop we keep the effort minimal — deep reasoning per turn slows the loop
    # without obvious wins, and tool dispatch already gives the model multiple
    # cheap turns to refine its answer.
    _REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")
    _DEFAULT_REASONING_EFFORT = "low"

    def __init__(self):
        try:
            from openai import AsyncOpenAI
            import tiktoken
        except ImportError as e:
            raise ImportError(
                "OpenAI provider requires `pip install openai tiktoken`"
            ) from e
        # AsyncOpenAI so the Responses SSE iteration is await-friendly and
        # plays nicely with concurrent asyncio tasks (matches AnthropicProvider).
        self.client = AsyncOpenAI()  # reads OPENAI_API_KEY from env
        self._tiktoken = tiktoken

    def _is_reasoning_model(self, model: str) -> bool:
        return any(model.startswith(p) for p in self._REASONING_PREFIXES)

    def format_tools(self, tools):
        # Responses API uses a flat tool definition — no `function` wrapper.
        return [
            {
                "type": "function",
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            }
            for t in tools
        ]

    def _to_responses_input(self, messages: list) -> list:
        """Translate canonical (Anthropic-shaped) messages into the Responses
        `input` array. Each Anthropic content block becomes its own item:
          - text blocks → {"role": "assistant", "content": "..."} messages
          - tool_use   → {"type": "function_call", "call_id", "name", "arguments"}
          - tool_result (user-role list content) →
              {"type": "function_call_output", "call_id", "output"}

        Order within a single assistant turn is preserved by flushing pending
        text whenever a tool_use appears, so interleaved text/tool sequences
        replay back to the model in the same order the model produced them.
        """
        out = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user" and isinstance(content, str):
                out.append({"role": "user", "content": content})
            elif role == "assistant":
                text_parts: list[str] = []
                for block in content:
                    btype = block.get("type")
                    if btype == "text":
                        text_parts.append(block["text"])
                    elif btype == "tool_use":
                        if text_parts:
                            out.append({
                                "role": "assistant",
                                "content": "".join(text_parts),
                            })
                            text_parts = []
                        out.append({
                            "type": "function_call",
                            "call_id": block["id"],
                            "name": block["name"],
                            "arguments": json.dumps(block["input"]),
                        })
                if text_parts:
                    out.append({
                        "role": "assistant",
                        "content": "".join(text_parts),
                    })
            elif role == "user" and isinstance(content, list):
                # tool_result batch: each becomes its own function_call_output
                for block in content:
                    if block.get("type") == "tool_result":
                        out.append({
                            "type": "function_call_output",
                            "call_id": block["tool_use_id"],
                            "output": block["content"],
                        })
        return out

    def _build_kwargs(self, messages, system, model, max_tokens) -> dict:
        kwargs: dict = {
            "model": model,
            "input": self._to_responses_input(messages),
            "max_output_tokens": max_tokens,
            # We manage conversation state on the client side, so don't have
            # OpenAI retain anything server-side (cheaper + no data lingering).
            "store": False,
        }
        if system:
            kwargs["instructions"] = system
        if self._is_reasoning_model(model):
            kwargs["reasoning"] = {"effort": self._DEFAULT_REASONING_EFFORT}
        return kwargs

    def stream(self, *, messages, tools, system, model, max_tokens):
        kwargs = self._build_kwargs(messages, system, model, max_tokens)
        kwargs["stream"] = True
        if tools:
            kwargs["tools"] = self.format_tools(tools)
        return _OpenAIStream(self.client, kwargs)

    async def create(self, *, messages, system, model, max_tokens):
        kwargs = self._build_kwargs(messages, system, model, max_tokens)
        resp = await self.client.responses.create(**kwargs)
        return _responses_to_normalized(resp)

    async def count_tokens(self, *, messages, tools, system, model):
        # No official count_tokens endpoint — approximate with tiktoken. Good
        # enough for the auto-trim heuristic; function-tool overhead isn't
        # exact but stays within a reasonable margin.
        # tiktoken is pure CPU so this `async def` body is fully synchronous;
        # the async signature is for interface parity with AnthropicProvider.
        try:
            enc = self._tiktoken.encoding_for_model(model)
        except KeyError:
            enc = self._tiktoken.get_encoding("cl100k_base")
        total = 0
        if system:
            total += len(enc.encode(system)) + 4
        for item in self._to_responses_input(messages):
            it = item.get("type")
            if it == "function_call":
                total += len(enc.encode(item.get("name", "")))
                total += len(enc.encode(item.get("arguments", "") or ""))
            elif it == "function_call_output":
                total += len(enc.encode(item.get("output", "") or ""))
            else:
                c = item.get("content")
                if isinstance(c, str):
                    total += len(enc.encode(c))
            total += 4  # per-item overhead
        for t in (tools or []):
            total += len(enc.encode(json.dumps(t)))
        return total


class Agent:
    """Multi-turn agent. Keeps message history across chat() calls."""

    def __init__(
        self,
        provider: LLMProvider,
        session: ClientSession,
        tools: list,
        mcp_tool_names: set,
        memory: MemoryStore | None = None,
        skills: SkillsRegistry | None = None,
        model: str | None = None,
        max_turns: int = 10,
        max_input_tokens: int = 100_000,
        keep_recent_turns: int = 5,
        system: str | None = DEFAULT_SYSTEM,
        approval_mode: str = "ask",
        silent: bool = False,
        depth: int = 0,
        hooks: dict | None = None,
    ):
        if approval_mode not in APPROVAL_MODES:
            raise ValueError(f"approval_mode must be one of {APPROVAL_MODES}")
        self.provider = provider
        self.session = session
        self.tools = tools
        self.mcp_tool_names = mcp_tool_names
        self.memory = memory
        self.skills = skills
        self.model = model or provider.default_model
        self.max_turns = max_turns
        self.max_input_tokens = max_input_tokens
        self.keep_recent_turns = keep_recent_turns
        self.system = system
        self.approval_mode = approval_mode
        self.silent = silent      # suppress streaming/tool prints (sub-agents)
        self.depth = depth        # 0 = user-facing; >0 = spawned by another agent
        self.messages: list = []
        self.last_input_tokens: int = 0  # from most recent response.usage
        # Lifecycle hooks. Sub-agents do NOT inherit by design — hooks are the
        # parent session's observers; spawn_subagent always builds a child with
        # `hooks=None`, so the child starts with empty lists.
        self.hooks: dict[str, list] = {e: [] for e in HOOK_EVENTS}
        if hooks:
            for event, fns in hooks.items():
                if event not in HOOK_EVENTS:
                    raise ValueError(
                        f"unknown hook event: {event!r}; one of {HOOK_EVENTS}"
                    )
                for fn in (fns if isinstance(fns, list) else [fns]):
                    self.hooks[event].append(fn)

    def on(self, event: str, fn) -> None:
        """Register `fn` as a callback for the lifecycle event `event`.
        See `HOOK_EVENTS` for valid event names and their callback signatures.
        Multiple hooks per event run in registration (FIFO) order."""
        if event not in HOOK_EVENTS:
            raise ValueError(
                f"unknown hook event: {event!r}; one of {HOOK_EVENTS}"
            )
        self.hooks[event].append(fn)

    def _fire(self, event: str, *args, **kwargs) -> None:
        """Run all hooks registered for `event` in FIFO order. Hook exceptions
        are swallowed (and logged unless silent) so a buggy hook can't crash
        the agent — except `HookBlocked`, which propagates so `_dispatch_tool`
        can route it back to the model as a synthetic tool result."""
        for fn in self.hooks.get(event, ()):
            try:
                fn(*args, **kwargs)
            except HookBlocked:
                raise
            except Exception as e:
                if not self.silent:
                    print(f"  [hook error] {event}: {type(e).__name__}: {e}")

    @staticmethod
    def _describe_block(block: dict) -> str:
        """Return a short label for one content block."""
        btype = block.get("type")
        if btype == "text":
            return "text"
        if btype == "tool_use":
            return f"tool_use:{block.get('name')}"
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

    async def count_tokens(self) -> int:
        """How many input tokens our current messages+tools+system use.
        Exact for Anthropic (API endpoint); approximated via tiktoken for OpenAI."""
        if not self.messages:
            return 0
        return await self.provider.count_tokens(
            messages=self.messages,
            tools=self.tools,
            system=self.system,
            model=self.model,
        )

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
        response = await self.provider.create(
            messages=prompt,
            system=None,
            model=self.model,
            max_tokens=2048,
        )
        return "".join(b["text"] for b in response.content if b["type"] == "text")

    async def _trim_to_budget(self) -> int:
        """Roll oldest turns into a summary message until under the input-token
        budget. Keeps the last `keep_recent_turns` turns verbatim. Returns the
        number of turns folded into the summary."""
        if not self.messages or await self.count_tokens() <= self.max_input_tokens:
            return 0

        folded_total = 0
        # One pass is normally enough: after folding, the next iteration has
        # exactly keep_recent_turns real turns and excess becomes 0. The while
        # is here for safety, not repeated folding.
        while await self.count_tokens() > self.max_input_tokens:
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

    async def _approve_run_shell(self, command: str) -> tuple[bool, str | None]:
        """Decide whether to run a shell command. Returns (approved, decline_msg).
        May mutate self.approval_mode if the user picks 'always'."""
        if self.approval_mode == "auto":
            return True, None
        if self.approval_mode == "safe" and is_safe_shell(command):
            return True, None
        # ask mode, or safe mode falling through to manual approval
        print(f"\n  [shell] Approve this command?")
        print(f"    > {command}")
        choice = (await asyncio.to_thread(
            input, "    [y]es / [n]o / [a]lways (this session) > "
        )).strip().lower()
        if choice in ("y", "yes", ""):
            return True, None
        if choice in ("a", "always"):
            self.approval_mode = "auto"
            print("  [shell] mode -> auto (no further prompts this session)")
            return True, None
        return False, f"User declined to run: {command}. Suggest an alternative or stop."

    async def _spawn_subagent(
        self, task: str, allowed_tools: list[str] | None = None
    ) -> str:
        """Spawn a depth+1 sub-agent to handle `task`. Sub-agent has its own
        context (no parent history), inherits approval mode, cannot recurse."""
        if self.depth + 1 >= MAX_DEPTH:
            return (
                f"ERROR: max sub-agent depth ({MAX_DEPTH}) reached. "
                "Do this task yourself instead of spawning."
            )
        # Strip spawn_agent (no recursion) and remember (read-only memory).
        forbidden = {"spawn_agent", "remember"}
        child_tools = [t for t in self.tools if t["name"] not in forbidden]
        if allowed_tools is not None:
            allow = set(allowed_tools) - forbidden
            child_tools = [t for t in child_tools if t["name"] in allow]
        child_mcp_names = {
            t["name"] for t in child_tools if t["name"] in self.mcp_tool_names
        }

        preview = task[:60] + ("..." if len(task) > 60 else "")
        print(f"\n  [subagent] depth={self.depth + 1} spawning: {preview}")

        # Sub-agent gets the same skills addendum so it can `load_skill` too.
        child_system = SUBAGENT_SYSTEM
        if self.skills:
            child_system += self.skills.system_addendum()

        child = Agent(
            provider=self.provider,
            session=self.session,
            tools=child_tools,
            mcp_tool_names=child_mcp_names,
            memory=self.memory,  # shared store; remember tool already stripped
            skills=self.skills,
            model=self.model,
            max_turns=8,
            max_input_tokens=self.max_input_tokens,
            keep_recent_turns=self.keep_recent_turns,
            system=child_system,
            approval_mode=self.approval_mode,
            silent=True,
            depth=self.depth + 1,
        )
        try:
            answer = await child.chat(task)
        except Exception as e:
            print(f"  [subagent] failed: {e}")
            return f"ERROR: sub-agent failed: {e}"
        n_assistant_turns = sum(
            1 for m in child.messages if m["role"] == "assistant"
        )
        print(
            f"  [subagent] done "
            f"(turns={n_assistant_turns}, last_input_tokens={child.last_input_tokens})"
        )
        return answer

    async def _dispatch_tool(self, name: str, args: dict) -> str:
        try:
            self._fire("pre_tool", name, args)
        except HookBlocked as e:
            self._fire("post_tool", name, args, e.result, e)
            return e.result

        # Approval gate: run_shell needs explicit approval before dispatch;
        # if declined, short-circuit with the decline message.
        if name == "run_shell":
            approved, decline_msg = await self._approve_run_shell(args["command"])
            if not approved:
                result = decline_msg or "User declined."
                self._fire("post_tool", name, args, result, None)
                return result

        # Dispatch (run_shell falls through here once approved).
        if name == "spawn_agent":
            result = await self._spawn_subagent(
                args["task"], args.get("allowed_tools")
            )
        elif name in self.mcp_tool_names:
            result = await call_mcp_tool(self.session, name, args)
        else:
            result = execute_native_tool(
                name, args, memory=self.memory, skills=self.skills
            )

        self._fire("post_tool", name, args, result, None)
        return result

    async def chat(self, user_message: str) -> str:
        """Send one user message, run the tool-use loop, stream output to stdout
        (unless `self.silent`), and return the final reply text."""
        self.messages.append({"role": "user", "content": user_message})
        self._fire("user_message", user_message)
        folded = await self._trim_to_budget()
        if folded and not self.silent:
            print(f"  [trim] rolled {folded} oldest turn(s) into summary to stay "
                  f"under {self.max_input_tokens} tokens")

        for turn_idx in range(self.max_turns):
            if not self.silent:
                print("claude> ", end="", flush=True)

            self._fire("pre_turn", turn_idx, self.messages)

            text_emitted = False
            async with self.provider.stream(
                messages=self.messages,
                tools=self.tools,
                system=self.system,
                model=self.model,
                max_tokens=1024,
            ) as stream:
                async for chunk in stream.text_stream:
                    if not self.silent:
                        print(chunk, end="", flush=True)
                    self._fire("text_chunk", chunk)
                    text_emitted = True
                response = await stream.get_final_message()
            if text_emitted and not self.silent:
                print()  # terminate the streamed line

            self.last_input_tokens = response.usage.input_tokens
            self.messages.append({"role": "assistant", "content": response.content})
            self._fire("post_turn", turn_idx, response, response.usage)

            if response.stop_reason == "end_turn":
                final_text = "".join(
                    b["text"] for b in response.content if b["type"] == "text"
                )
                self._fire("assistant_message", final_text)
                return final_text

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block["type"] == "tool_use":
                        b_name, b_input, b_id = block["name"], block["input"], block["id"]
                        if not self.silent:
                            source = "mcp" if b_name in self.mcp_tool_names else "native"
                            print(f"  [{source} tool] {b_name}({b_input})")
                        result = await self._dispatch_tool(b_name, b_input)
                        if not self.silent:
                            preview = result[:100] + ("..." if len(result) > 100 else "")
                            print(f"  [tool result] {preview} \n  [/tool result] ")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": b_id,
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
    parser.add_argument(
        "--system-file",
        help="Path to a file whose contents replace the built-in system prompt",
    )
    parser.add_argument(
        "--no-system", action="store_true",
        help="Disable the system prompt entirely (run model in raw mode)",
    )
    parser.add_argument(
        "--approval", choices=APPROVAL_MODES, default="ask",
        help=("run_shell approval mode: 'auto' = no gating, "
              "'ask' = prompt every command (default), "
              "'safe' = auto-approve a read-only allow-list, prompt others"),
    )
    parser.add_argument(
        "--yolo", action="store_true",
        help="Shortcut for --approval auto (you accept all consequences)",
    )
    parser.add_argument(
        "--trace", action="store_true",
        help="Print every lifecycle event (hooks demo / debug)",
    )
    parser.add_argument(
        "--provider", choices=("anthropic", "openai"), default="anthropic",
        help="LLM provider to use (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        help="Override the provider's default model "
             f"(anthropic={ANTHROPIC_MODEL}, openai={OPENAI_MODEL})",
    )
    parser.add_argument(
        "--memory", choices=MEMORY_BACKENDS, default="jsonl",
        help=("long-term memory backend: 'jsonl' = local append-only file with "
              "in-memory cosine search (default), 'pgvector' = Postgres + "
              "pgvector with an HNSW index"),
    )
    parser.add_argument(
        "--pg-dsn",
        help="Postgres DSN for --memory pgvector "
             "(default: DATABASE_URL env var)",
    )
    cli_args = parser.parse_args()
    if cli_args.yolo:
        cli_args.approval = "auto"

    if cli_args.provider == "openai":
        provider: LLMProvider = OpenAIProvider()
    else:
        provider = AnthropicProvider()
    model = cli_args.model or provider.default_model

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
                if cli_args.memory == "pgvector":
                    memory = PgVectorStore(dsn=cli_args.pg_dsn)
                    mem_status = f"pgvector ({memory.count()} records)"
                else:
                    memory = MemoryStore()
                    mem_status = f"jsonl ({len(memory.records)} records loaded)"
            except Exception as e:
                memory = None
                mem_status = f"disabled ({e})"

            skills = SkillsRegistry()
            skill_status = (
                f"{len(skills.skills)} loaded ({', '.join(skills.skills.keys())})"
                if skills.skills else f"none found in {SKILLS_DIR}/"
            )

            if cli_args.no_system:
                system = None
                sys_status = "disabled"
            elif cli_args.system_file:
                system = Path(cli_args.system_file).read_text(encoding="utf-8")
                sys_status = f"loaded from {cli_args.system_file} ({len(system)} chars)"
            else:
                system = DEFAULT_SYSTEM
                sys_status = f"built-in default ({len(system)} chars)"

            # Append skill names + one-line descriptions to whatever system
            # prompt we ended up with (default, file, or even raw if not None).
            if system is not None and skills.skills:
                system = system + "\n" + skills.system_addendum()

            print(f"[init] provider:     {provider.name} (model={model})")
            print(f"[init] native tools: {[t['name'] for t in NATIVE_TOOLS]}")
            print(f"[init] mcp tools:    {sorted(mcp_tool_names)}")
            print(f"[init] memory:       {mem_status}")
            print(f"[init] skills:       {skill_status}")
            print(f"[init] system:       {sys_status}")
            print(f"[init] approval:     {cli_args.approval}"
                  f"{' (run_shell gated)' if cli_args.approval != 'auto' else ' (no gating)'}")
            print("[hint] /save /load /list /tokens /compact /messages /memories /skills /system /reset /exit\n")

            agent = Agent(
                provider=provider,
                session=session,
                tools=all_tools,
                mcp_tool_names=mcp_tool_names,
                memory=memory,
                skills=skills,
                model=model,
                max_input_tokens=cli_args.max_input_tokens,
                keep_recent_turns=cli_args.keep_recent_turns,
                system=system,
                approval_mode=cli_args.approval,
            )

            if cli_args.trace:
                # Demo: log every lifecycle event to stdout. Hooks are pure
                # observation here — they don't mutate args or block calls.
                def on_user(text):
                    print(f"  [hook] user_message ({len(text)} chars)", flush=True)
                def on_pre_turn(turn_idx, messages):
                    print(f"  [hook] pre_turn turn={turn_idx} "
                          f"msgs={len(messages)}", flush=True)
                def on_post_turn(turn_idx, response, usage):
                    print(f"  [hook] post_turn turn={turn_idx} "
                          f"in_tok={usage.input_tokens} "
                          f"out_tok={usage.output_tokens}", flush=True)
                def on_pre_tool(name, args):
                    print(f"  [hook] pre_tool {name}", flush=True)
                def on_post_tool(name, args, result, error):
                    tag = "BLOCKED" if error else "ok"
                    print(f"  [hook] post_tool {name} "
                          f"({tag}, {len(result)} chars)", flush=True)
                def on_assistant(text):
                    print(f"  [hook] assistant_message ({len(text)} chars)",
                          flush=True)
                agent.on("user_message", on_user)
                agent.on("pre_turn", on_pre_turn)
                agent.on("post_turn", on_post_turn)
                agent.on("pre_tool", on_pre_tool)
                agent.on("post_tool", on_post_tool)
                agent.on("assistant_message", on_assistant)
                print("[init] trace:       on (lifecycle events will be logged)")

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
                    n = await agent.count_tokens()
                    print(f"[tokens] {n} / {agent.max_input_tokens} "
                          f"({len(agent.messages)} messages)\n")
                    continue
                if user_in == "/messages":
                    agent.dump()
                    print()
                    continue
                if user_in == "/system":
                    if agent.system is None:
                        print("[no system prompt]\n")
                    else:
                        print(f"--- system ({len(agent.system)} chars) ---")
                        print(agent.system.rstrip())
                        print("--- end ---\n")
                    continue
                if user_in == "/skills":
                    items = agent.skills.list() if agent.skills else []
                    if not items:
                        print("  (no skills)\n")
                    else:
                        for it in items:
                            print(f"  {it['name']}: {it['description']}")
                        print()
                    continue
                if user_in == "/memories":
                    if agent.memory is None:
                        print("[memory disabled]\n")
                    else:
                        items = agent.memory.all()
                        if not items:
                            print("  (empty)\n")
                        else:
                            for it in items:
                                tags = f" {it['tags']}" if it["tags"] else ""
                                print(f"  {it['id']}{tags}: {it['text']}")
                            print()
                    continue
                if user_in == "/compact":
                    before = await agent.count_tokens()
                    summary = await agent.compact()
                    after = await agent.count_tokens()
                    print(f"[compacted {before} -> {after} tokens]")
                    print(f"[summary] {summary[:300]}{'...' if len(summary) > 300 else ''}\n")
                    continue

                print()  # blank line before assistant output
                await agent.chat(user_in)
                print(f"[tokens: {agent.last_input_tokens}]\n")


if __name__ == "__main__":
    asyncio.run(main())
