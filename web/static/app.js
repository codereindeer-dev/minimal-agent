const chatEl = document.getElementById("chat");
const formEl = document.getElementById("input-form");
const inputEl = document.getElementById("input");
const btnEl = formEl.querySelector("button");
const statusEl = document.getElementById("status");

function addBubble(role, text) {
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  div.textContent = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

async function send(text) {
  addBubble("user", text);
  const pending = addBubble("assistant", "…");
  btnEl.disabled = true;
  statusEl.textContent = "thinking…";
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    pending.textContent = data.reply || "(no reply)";
    statusEl.textContent = data.tokens ? `${data.tokens} tokens` : "";
  } catch (e) {
    pending.textContent = "Error: " + e.message;
    pending.classList.add("error");
    statusEl.textContent = "";
  } finally {
    btnEl.disabled = false;
    inputEl.focus();
  }
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
