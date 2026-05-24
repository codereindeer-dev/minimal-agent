const chatEl = document.getElementById("chat");
const formEl = document.getElementById("input-form");
const inputEl = document.getElementById("input");
const btnEl = formEl.querySelector("button");
const statusEl = document.getElementById("status");

let activeAssistantBubble = null;
const toolCards = new Map(); // tool_call_id -> { card, resultEl }

function addUserBubble(text) {
  const div = document.createElement("div");
  div.className = "bubble user";
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

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function scrollDown() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function setStatus(text) {
  statusEl.textContent = text;
}

async function send(text) {
  addUserBubble(text);
  btnEl.disabled = true;
  setStatus("thinking…");

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
    btnEl.disabled = false;
    setStatus("");
    return;
  }

  const es = new EventSource(`/api/stream?request_id=${requestId}`);
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    switch (ev.type) {
      case "text": {
        const b = ensureAssistantBubble();
        b.textContent += ev.chunk;
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
        setStatus(`${ev.input} in / ${ev.output} out`);
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
        btnEl.disabled = false;
        inputEl.focus();
        break;
    }
  };
  es.onerror = () => {
    es.close();
    finalizeAssistantBubble();
    btnEl.disabled = false;
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
