const chatEl = document.getElementById("chat");
const formEl = document.getElementById("input-form");
const inputEl = document.getElementById("input");
const btnEl = formEl.querySelector("button");
const statusEl = document.getElementById("status");

function addBubble(role) {
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

function setStatus(text) {
  statusEl.textContent = text;
}

async function send(text) {
  const userBubble = addBubble("user");
  userBubble.textContent = text;

  let bubble = addBubble("assistant");
  bubble.textContent = "";
  bubble.classList.add("streaming");

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
    bubble.textContent = "Error: " + e.message;
    bubble.classList.add("error");
    btnEl.disabled = false;
    setStatus("");
    return;
  }

  const es = new EventSource(`/api/stream?request_id=${requestId}`);
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.type === "text") {
      bubble.textContent += ev.chunk;
      chatEl.scrollTop = chatEl.scrollHeight;
    } else if (ev.type === "assistant_done") {
      bubble.classList.remove("streaming");
      // text already accumulated from chunks; assistant_done is the
      // authoritative final string in case of any drift
      bubble.textContent = ev.text;
      // start a fresh bubble for the next turn (if tool calls cause another)
      bubble = addBubble("assistant");
      bubble.textContent = "";
      bubble.classList.add("streaming");
    } else if (ev.type === "usage") {
      setStatus(`${ev.input} in / ${ev.output} out`);
    } else if (ev.type === "error") {
      bubble.textContent = "Error: " + ev.message;
      bubble.classList.add("error");
    } else if (ev.type === "done") {
      es.close();
      if (!bubble.textContent) bubble.remove();
      btnEl.disabled = false;
      inputEl.focus();
    }
  };
  es.onerror = () => {
    es.close();
    if (!bubble.textContent) bubble.textContent = "(stream closed)";
    bubble.classList.add("error");
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
