// ─── element refs ────────────────────────────────────────────────────────────
const chatEl = document.getElementById("chat");
const formEl = document.getElementById("input-form");
const inputEl = document.getElementById("input");
const sendBtn = formEl.querySelector("button");
const tokenMeterEl = document.getElementById("token-meter");
const modelLineEl = document.getElementById("model-line");
const sessionListEl = document.getElementById("session-list");
const sessionNameEl = document.getElementById("session-name");
const sessionSaveBtn = document.getElementById("session-save");
const memoryListEl = document.getElementById("memory-list");
const memCountEl = document.getElementById("mem-count");
const skillListEl = document.getElementById("skill-list");
const skillCountEl = document.getElementById("skill-count");
const resetBtn = document.getElementById("btn-reset");
const compactBtn = document.getElementById("btn-compact");

// ─── chat state ──────────────────────────────────────────────────────────────
let activeAssistantBubble = null;
const toolCards = new Map();

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function scrollDown() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

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
  scrollDown();
}

function addAssistantBubbleFinal(text) {
  const div = document.createElement("div");
  div.className = "bubble assistant";
  div.textContent = text;
  chatEl.appendChild(div);
  scrollDown();
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

function addToolCard(toolCallId, name, args) {
  finalizeAssistantBubble();
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
  chatEl.appendChild(card);
  toolCards.set(toolCallId, {
    card,
    statusEl: card.querySelector(".tool-status"),
    resultEl: card.querySelector(".tool-result"),
  });
  scrollDown();
}

function fillToolResult(toolCallId, result, blocked) {
  const entry = toolCards.get(toolCallId);
  if (!entry) return;
  entry.statusEl.textContent = blocked ? "blocked" : "done";
  entry.statusEl.classList.add(blocked ? "blocked" : "done");
  entry.resultEl.textContent = result;
  entry.resultEl.classList.remove("hidden");
  toolCards.delete(toolCallId);
  scrollDown();
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
  scrollDown();
}

// ─── send / SSE ──────────────────────────────────────────────────────────────
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
      case "text": {
        ensureAssistantBubble().textContent += ev.chunk;
        scrollDown();
        break;
      }
      case "assistant_done":
        finalizeAssistantBubble(ev.text);
        break;
      case "tool_start":
        addToolCard(ev.tool_call_id, ev.name, ev.args);
        break;
      case "tool_end":
        fillToolResult(ev.tool_call_id, ev.result, ev.blocked);
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

// ─── sidebar: state / sessions / memories / skills ───────────────────────────
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
  modelLineEl.textContent = `${s.provider} · ${s.model}`;
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

async function refreshMemories() {
  const r = await fetch("/api/memories");
  const { enabled, memories } = await r.json();
  memoryListEl.innerHTML = "";
  if (!enabled) {
    memCountEl.textContent = "(disabled)";
    return;
  }
  memCountEl.textContent = `(${memories.length})`;
  if (!memories.length) {
    memoryListEl.innerHTML = `<li class="sb-empty">no memories yet</li>`;
    return;
  }
  for (const m of memories) {
    const li = document.createElement("li");
    li.className = "sb-item sb-memory";
    const tags = m.tags && m.tags.length ? ` [${m.tags.join(",")}]` : "";
    li.innerHTML = `<div class="sb-mem-id"></div><div class="sb-mem-text"></div>`;
    li.querySelector(".sb-mem-id").textContent = `#${m.id}${tags}`;
    li.querySelector(".sb-mem-text").textContent = m.text;
    memoryListEl.appendChild(li);
  }
}

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
    scrollDown();
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
