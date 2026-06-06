// ─── element refs ────────────────────────────────────────────────────────────
const chatEl = document.getElementById("chat");
const formEl = document.getElementById("input-form");
const inputEl = document.getElementById("input");
const sendBtn = formEl.querySelector("button");
const tokenMeterEl = document.getElementById("token-meter");
const providerSelectEl = document.getElementById("provider-select");
const modelInputEl = document.getElementById("model-input");
const providerApplyBtn = document.getElementById("provider-apply");
const sessionListEl = document.getElementById("session-list");
const sessionNameEl = document.getElementById("session-name");
const sessionSaveBtn = document.getElementById("session-save");
const memoryListEl = document.getElementById("memory-list");
const memCountEl = document.getElementById("mem-count");
const memBackendEl = document.getElementById("mem-backend");
const memSearchEl = document.getElementById("mem-search");
const memSearchBtn = document.getElementById("mem-search-btn");
const skillListEl = document.getElementById("skill-list");
const skillCountEl = document.getElementById("skill-count");
const resetBtn = document.getElementById("btn-reset");
const compactBtn = document.getElementById("btn-compact");

// ─── shared helpers ──────────────────────────────────────────────────────────
function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function scrollDownEl(el) {
  el.scrollTop = el.scrollHeight;
}

// ─── single-chat state ───────────────────────────────────────────────────────
let activeAssistantBubble = null;
const toolCards = new Map();
let memTouched = false;  // set when a remember/recall tool fired this turn

function clearChat() {
  chatEl.innerHTML = "";
  activeAssistantBubble = null;
  toolCards.clear();
}

function addUserBubble(text) {
  const div = document.createElement("div");
  div.className = "bubble user";
  div.textContent = text;
  chatEl.appendChild(div);
  scrollDownEl(chatEl);
}

function addAssistantBubbleFinal(text) {
  const div = document.createElement("div");
  div.className = "bubble assistant";
  div.textContent = text;
  chatEl.appendChild(div);
  scrollDownEl(chatEl);
}

function ensureAssistantBubble() {
  if (activeAssistantBubble) return activeAssistantBubble;
  const div = document.createElement("div");
  div.className = "bubble assistant streaming";
  div.textContent = "";
  chatEl.appendChild(div);
  activeAssistantBubble = div;
  return div;
}

function finalizeAssistantBubble(finalText) {
  if (!activeAssistantBubble) return;
  activeAssistantBubble.classList.remove("streaming");
  if (finalText !== undefined) activeAssistantBubble.textContent = finalText;
  if (!activeAssistantBubble.textContent) activeAssistantBubble.remove();
  activeAssistantBubble = null;
}

function addToolCard(parent, toolCallId, name, args, registry) {
  const card = document.createElement("div");
  card.className = "tool-card";
  card.innerHTML = `
    <div class="tool-header">
      <span class="tool-name">${escapeHtml(name)}</span>
      <span class="tool-status">running…</span>
    </div>
    <pre class="tool-args">${escapeHtml(JSON.stringify(args, null, 2))}</pre>
    <pre class="tool-result hidden"></pre>
  `;
  parent.appendChild(card);
  registry.set(toolCallId, {
    card,
    statusEl: card.querySelector(".tool-status"),
    resultEl: card.querySelector(".tool-result"),
  });
  scrollDownEl(parent);
}

function fillToolResult(toolCallId, result, blocked, registry) {
  const entry = registry.get(toolCallId);
  if (!entry) return;
  entry.statusEl.textContent = blocked ? "blocked" : "done";
  entry.statusEl.classList.add(blocked ? "blocked" : "done");
  entry.resultEl.textContent = result;
  entry.resultEl.classList.remove("hidden");
  registry.delete(toolCallId);
  scrollDownEl(entry.card.parentElement);
}

function addApprovalCard(approvalId, command) {
  finalizeAssistantBubble();
  const card = document.createElement("div");
  card.className = "approval-card";
  card.innerHTML = `
    <div class="approval-header">⚠ run_shell approval required</div>
    <pre class="approval-cmd"></pre>
    <div class="approval-buttons">
      <button class="approve">Approve</button>
      <button class="deny">Deny</button>
    </div>
  `;
  card.querySelector(".approval-cmd").textContent = command;
  const approveBtn = card.querySelector(".approve");
  const denyBtn = card.querySelector(".deny");
  const decide = async (decision) => {
    approveBtn.disabled = true;
    denyBtn.disabled = true;
    await fetch("/api/approve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ approval_id: approvalId, decision }),
    });
    card.classList.add(decision === "approve" ? "approved" : "denied");
    card.querySelector(".approval-header").textContent =
      decision === "approve" ? "✓ Approved" : "✕ Denied";
  };
  approveBtn.addEventListener("click", () => decide("approve"));
  denyBtn.addEventListener("click", () => decide("deny"));
  chatEl.appendChild(card);
  scrollDownEl(chatEl);
}

// ─── send / SSE for single chat ──────────────────────────────────────────────
async function send(text) {
  addUserBubble(text);
  sendBtn.disabled = true;

  let requestId;
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    requestId = (await res.json()).request_id;
  } catch (e) {
    const err = document.createElement("div");
    err.className = "bubble assistant error";
    err.textContent = "Error: " + e.message;
    chatEl.appendChild(err);
    sendBtn.disabled = false;
    return;
  }

  const es = new EventSource(`/api/stream?request_id=${requestId}`);
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    switch (ev.type) {
      case "text":
        ensureAssistantBubble().textContent += ev.chunk;
        scrollDownEl(chatEl);
        break;
      case "assistant_done":
        finalizeAssistantBubble(ev.text);
        break;
      case "tool_start":
        finalizeAssistantBubble();
        addToolCard(chatEl, ev.tool_call_id, ev.name, ev.args, toolCards);
        break;
      case "tool_end":
        fillToolResult(ev.tool_call_id, ev.result, ev.blocked, toolCards);
        if (!ev.blocked && (ev.name === "remember" || ev.name === "recall")) {
          memTouched = true;
        }
        break;
      case "approval_request":
        addApprovalCard(ev.approval_id, ev.command);
        break;
      case "usage":
        renderTokenMeter(ev.input);
        break;
      case "error": {
        const b = ensureAssistantBubble();
        b.textContent = "Error: " + ev.message;
        b.classList.add("error");
        finalizeAssistantBubble();
        break;
      }
      case "done":
        finalizeAssistantBubble();
        es.close();
        sendBtn.disabled = false;
        inputEl.focus();
        refreshState();
        if (memTouched) {
          memTouched = false;
          refreshMemories({ highlight: true });
        }
        break;
    }
  };
  es.onerror = () => {
    es.close();
    finalizeAssistantBubble();
    sendBtn.disabled = false;
  };
}

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = "";
  send(text);
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    formEl.requestSubmit();
  }
});

// ─── sidebar: state / sessions / memories / skills / provider ────────────────
let maxTokens = 100000;

function renderTokenMeter(tokens) {
  const pct = Math.min(100, Math.round((tokens / maxTokens) * 100));
  tokenMeterEl.innerHTML = `
    <span class="tk-num">${tokens.toLocaleString()}</span>
    <span class="tk-sep">/</span>
    <span class="tk-max">${maxTokens.toLocaleString()}</span>
    <div class="tk-bar"><div class="tk-fill" style="width:${pct}%"></div></div>
  `;
}

function renderMessagesFromHistory(messages) {
  clearChat();
  for (const m of messages) {
    if (m.role === "user") addUserBubble(m.text);
    else if (m.role === "assistant") addAssistantBubbleFinal(m.text);
  }
}

async function refreshState() {
  const r = await fetch("/api/state");
  const s = await r.json();
  maxTokens = s.max_tokens;
  providerSelectEl.value = s.provider;
  modelInputEl.value = s.model;
  renderTokenMeter(s.tokens);
}

async function refreshSessions() {
  const r = await fetch("/api/sessions");
  const { sessions } = await r.json();
  sessionListEl.innerHTML = "";
  if (!sessions.length) {
    sessionListEl.innerHTML = `<li class="sb-empty">no sessions saved</li>`;
    return;
  }
  for (const name of sessions) {
    const li = document.createElement("li");
    li.className = "sb-item";
    li.innerHTML = `
      <span class="sb-item-name"></span>
      <span class="sb-item-actions">
        <button class="sb-load">Load</button>
        <button class="sb-del">×</button>
      </span>
    `;
    li.querySelector(".sb-item-name").textContent = name;
    li.querySelector(".sb-load").addEventListener("click", () => loadSession(name));
    li.querySelector(".sb-del").addEventListener("click", () => deleteSession(name));
    sessionListEl.appendChild(li);
  }
}

async function saveSession() {
  const name = sessionNameEl.value.trim();
  if (!name) return;
  await fetch("/api/sessions/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  sessionNameEl.value = "";
  refreshSessions();
}

async function loadSession(name) {
  const r = await fetch("/api/sessions/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!r.ok) return;
  const { messages } = await r.json();
  renderMessagesFromHistory(messages);
  refreshState();
}

async function deleteSession(name) {
  if (!confirm(`Delete session "${name}"?`)) return;
  await fetch(`/api/sessions/${encodeURIComponent(name)}`, { method: "DELETE" });
  refreshSessions();
}

const BACKEND_LABELS = { MemoryStore: "jsonl", PgVectorStore: "pgvector" };

// Build one memory <li>. `score` (0–1) is rendered only for search results.
function renderMemoryItem(m, { score = null, highlight = false } = {}) {
  const li = document.createElement("li");
  li.className = "sb-item sb-memory" + (highlight ? " sb-flash" : "");
  const scoreHtml = score !== null
    ? `<span class="sb-mem-score" title="cosine similarity">${score.toFixed(3)}</span>`
    : "";
  li.innerHTML =
    `<div class="sb-mem-head">` +
      `<span class="sb-mem-id"></span>` +
      scoreHtml +
      `<button class="sb-del" title="Forget this memory">✕</button>` +
    `</div>` +
    `<div class="sb-mem-text"></div>` +
    `<div class="sb-mem-tags"></div>`;
  li.querySelector(".sb-mem-id").textContent = `#${m.id}`;
  li.querySelector(".sb-mem-text").textContent = m.text;
  const tagsEl = li.querySelector(".sb-mem-tags");
  if (m.tags && m.tags.length) {
    for (const t of m.tags) {
      const chip = document.createElement("span");
      chip.className = "sb-tag";
      chip.textContent = t;
      tagsEl.appendChild(chip);
    }
  } else {
    tagsEl.remove();
  }
  li.querySelector(".sb-del").addEventListener("click", () => deleteMemory(m.id));
  return li;
}

async function refreshMemories({ highlight = false } = {}) {
  const r = await fetch("/api/memories");
  const { enabled, backend, count, memories } = await r.json();
  memoryListEl.innerHTML = "";
  if (!enabled) {
    memCountEl.textContent = "(disabled)";
    memBackendEl.textContent = "";
    return;
  }
  memCountEl.textContent = `(${count})`;
  memBackendEl.textContent = BACKEND_LABELS[backend] || backend || "";
  if (!memories.length) {
    memoryListEl.innerHTML = `<li class="sb-empty">no memories yet</li>`;
    return;
  }
  for (const m of memories) {
    memoryListEl.appendChild(renderMemoryItem(m, { highlight }));
  }
}

async function searchMemories() {
  const q = memSearchEl.value.trim();
  if (!q) {
    refreshMemories();
    return;
  }
  memSearchBtn.disabled = true;
  memoryListEl.innerHTML = `<li class="sb-empty sb-loading">searching…</li>`;
  try {
    const r = await fetch(`/api/memories/search?q=${encodeURIComponent(q)}`);
    const { enabled, results } = await r.json();
    memoryListEl.innerHTML = "";
    if (!enabled) {
      memoryListEl.innerHTML = `<li class="sb-empty">memory disabled</li>`;
      return;
    }
    if (!results.length) {
      memoryListEl.innerHTML = `<li class="sb-empty">no matches</li>`;
      return;
    }
    for (const m of results) {
      memoryListEl.appendChild(renderMemoryItem(m, { score: m.score }));
    }
  } finally {
    memSearchBtn.disabled = false;
  }
}

async function deleteMemory(id) {
  if (!confirm(`Forget memory #${id}?`)) return;
  const r = await fetch(`/api/memories/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!r.ok) return;
  // Re-run the active search if there's a query, else refresh the full list.
  if (memSearchEl.value.trim()) searchMemories();
  else refreshMemories();
}

memSearchBtn.addEventListener("click", searchMemories);
memSearchEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); searchMemories(); }
});

async function refreshSkills() {
  const r = await fetch("/api/skills");
  const { skills } = await r.json();
  skillListEl.innerHTML = "";
  skillCountEl.textContent = `(${skills.length})`;
  if (!skills.length) {
    skillListEl.innerHTML = `<li class="sb-empty">no skills loaded</li>`;
    return;
  }
  for (const s of skills) {
    const li = document.createElement("li");
    li.className = "sb-item sb-skill";
    li.innerHTML = `<div class="sb-sk-name"></div><div class="sb-sk-desc"></div>`;
    li.querySelector(".sb-sk-name").textContent = s.name;
    li.querySelector(".sb-sk-desc").textContent = s.description;
    skillListEl.appendChild(li);
  }
}

async function applyProvider() {
  const provider = providerSelectEl.value;
  const model = modelInputEl.value.trim();
  providerApplyBtn.disabled = true;
  try {
    const r = await fetch("/api/provider", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider, model: model || null }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const s = await r.json();
    modelInputEl.value = s.model;
  } finally {
    providerApplyBtn.disabled = false;
  }
}

providerApplyBtn.addEventListener("click", applyProvider);
providerSelectEl.addEventListener("change", () => { modelInputEl.value = ""; });

resetBtn.addEventListener("click", async () => {
  if (!confirm("Clear the conversation history?")) return;
  await fetch("/api/reset", { method: "POST" });
  clearChat();
  refreshState();
});

compactBtn.addEventListener("click", async () => {
  compactBtn.disabled = true;
  compactBtn.textContent = "Compacting…";
  try {
    const r = await fetch("/api/compact", { method: "POST" });
    const { before, after } = await r.json();
    const note = document.createElement("div");
    note.className = "bubble assistant compact-note";
    note.textContent = `[compacted ${before} → ${after} tokens]`;
    chatEl.appendChild(note);
    scrollDownEl(chatEl);
    refreshState();
  } finally {
    compactBtn.disabled = false;
    compactBtn.textContent = "Compact";
  }
});

sessionSaveBtn.addEventListener("click", saveSession);
sessionNameEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    saveSession();
  }
});

// ─── boot ────────────────────────────────────────────────────────────────────
refreshState();
refreshSessions();
refreshMemories();
refreshSkills();
