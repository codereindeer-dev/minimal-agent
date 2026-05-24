# minimal-agent

從零手寫的 AI agent loop（核心單檔 Python）+ 一個可選的 peer-to-peer multi-agent group chat demo。可換 LLM provider（Anthropic / OpenAI）+ native tools + MCP + 長期記憶 + sub-agent 委派 + peer-to-peer group chat + lifecycle hooks + skills。沒有 LangChain、LlamaIndex、AutoGen 之類的 framework。

---

## 它是什麼

一個檔案（`minimal_agent.py`）做完這些事：

- 跑 LLM 的 tool-use loop（request → tool call → tool result → 迴圈）
- LLM provider 抽象：`--provider {anthropic,openai}` 一個旗標切換，Agent 主迴圈跨家共用
- 串流（streaming）輸出 assistant 訊息到 stdout（兩家 provider 都支援）
- 7 個原生工具：`run_shell` / `read_file` / `write_file` / `remember` / `recall` / `spawn_agent` / `load_skill`
- 接 MCP server（`mcp-server-fetch`），把它的工具跟原生工具混在一起讓模型用
- 對話持久化（JSON 序列化、跨 session 接續，跨 provider 也能 load）
- Context 管理：token 計算、自動 trim（rolling summary）、手動 `/compact`
- 長期記憶：Voyage AI embedding + JSONL append-only + cosine 相似度檢索
- 子 agent 委派：`spawn_agent` 工具、depth 限制 ≤ 2、context 隔離（自動繼承 provider）
- 6 個生命週期事件給外部 hook 註冊（`user_message`、`pre/post_turn`、`pre/post_tool`、`assistant_message`）
- Skills（`skills/<name>/SKILL.md`）：啟動時掃描，只把 name+description 注入 system prompt，模型自己決定何時 `load_skill` 載入完整指令

每個 git commit 都是一個可獨立 checkout 閱讀的小段。Clone 後 `git checkout` 早期 commit，從最簡單的版本一路讀上來。

---

## Quickstart

```bash
pip install anthropic python-dotenv mcp mcp-server-fetch voyageai
# 想用 OpenAI 再裝這兩個：
pip install openai tiktoken
# 想開 Web UI 再裝這兩個：
pip install fastapi uvicorn
```

建立 `.env`：

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...     # 只有 --provider openai 才需要
VOYAGE_API_KEY=...        # 只有 remember / recall 需要
```

跑（預設 Anthropic）：

```bash
python minimal_agent.py
```

切到 OpenAI：

```bash
python minimal_agent.py --provider openai           # 預設 gpt-5.5
python minimal_agent.py --provider openai --model gpt-4.1   # 覆蓋 model
```

```
you> 看一下現在目錄有什麼
claude> 我用 ls 看看。
  [native tool] run_shell({'command': 'ls'})
  [shell] Approve this command?
    > ls
    [y]es / [n]o / [a]lways (this session) > y
  [tool result] memory  minimal_agent.py  sessions  skills ...
claude> 目錄裡有 5 個項目：...
[tokens: 2341]
```

---

## CLI 參數

| 參數 | 作用 |
|------|------|
| `--provider {anthropic,openai}` | 選 LLM provider（預設 anthropic）|
| `--model <name>` | 覆蓋 provider 預設 model（anthropic=`claude-sonnet-4-6`、openai=`gpt-5.5`）|
| `--resume <name>` | 開機時載入指定的對話 session |
| `--max-input-tokens N` | input token 超過此值就自動 trim（預設 100k）|
| `--keep-recent-turns N` | trim 時保留最新 N 輪不壓縮（預設 5）|
| `--system-file <path>` | 用檔案內容當 system prompt |
| `--no-system` | 完全停用 system prompt |
| `--approval {auto,ask,safe}` | `run_shell` 批准模式：全自動 / 每次問 / 唯讀白名單自動 |
| `--yolo` | `--approval auto` 的捷徑 |
| `--trace` | 把所有 lifecycle 事件印出來（hooks demo / debug） |

---

## REPL slash commands

| 指令 | 作用 |
|------|------|
| `/save <name>` | 把目前對話存到 `sessions/<name>.json` |
| `/load <name>` | 載入存檔取代目前對話 |
| `/list` | 列出所有 session 存檔 |
| `/messages` | 印出目前對話歷史的逐筆摘要 |
| `/tokens` | 顯示目前 input token 用量 vs 上限 |
| `/compact` | 手動把整段歷史摘要成一條 recap |
| `/memories` | 列出所有長期記憶內容 |
| `/skills` | 列出 `skills/` 下所有 skill 的 name + description |
| `/system` | 印出目前 system prompt |
| `/reset` | 清空對話歷史 |
| `/exit` | 離開 |

---

## Peer-to-peer multi-agent: `group_chat.py`

建在 `minimal_agent.py` 之上的可選 demo，**core 一行不動**。`GroupChat` 編排 N 個 stateful Agent 輪流發言，每輪廣播 transcript 最後 N-1 條給下個 peer（round-robin 下剛好等於「我上次以來的全部」），`[DONE]` sentinel + max_rounds 兩層終止。Demo 是 planner + coder + reviewer 合作寫 + 互相 review，抓出 single agent 自己 review 看不到的 bug。

```bash
python group_chat.py "Write a Python fib(n) function plus a quick test."
```

關鍵設計：peer 跟 sub-agent 是兩種互斥的 multi-agent 模式 —— 所以 peer 的工具不能包含 `spawn_agent` 跟 `remember`，不然 model 會 spawn sub-agent 旁路掉整個 group chat。

---

## Web UI: `web/`

可選的瀏覽器介面，建在 `Agent.chat()` 之上。FastAPI + 原生 HTML/JS（沒有 React、沒有 build step）。

```bash
pip install fastapi uvicorn
uvicorn web.server:app --reload
# 開 http://localhost:8000
```

---

## 專案結構

```
minimal_agent.py    # Core agent loop，全部 single-agent 邏輯都在這
group_chat.py       # Peer-to-peer multi-agent demo（建在 minimal_agent.py 上面，core 不動）
web/                # 可選的 Web UI（FastAPI + 原生 HTML/JS）
skills/             # 每個子目錄一個 skill，內含 SKILL.md（frontmatter + 指令本文）
sessions/           # 對話存檔（已 gitignore）
memory/store.jsonl  # 長期記憶 append-only 檔（已 gitignore）
README.md           # 你正在讀這個
```

---

## Commit 演進

| Commit | 加了什麼 |
|--------|---------|
| `9729fa0` | 最初版：minimal loop + mock `get_weather` 工具 |
| `c1e9a04` | 把 mock 換成真的 `run_shell` / `read_file` / `write_file` |
| `8865be0` | 整合 MCP server，跟原生工具一起用 |
| `c18459c` | 重構成 `Agent` class + 多輪 REPL |
| `cc5718f` | Context 管理（`/tokens` `/compact` `/messages` + rolling summary auto-trim）|
| `8eac59c` | 對話持久化（`/save` `/load` `--resume`）|
| `d3528ad` | 長期記憶 via Voyage embedding（`remember` / `recall`）|
| `bffa463` | Token-by-token 串流輸出 |
| `2772931` | 預設 system prompt + `--system-file` / `--no-system` |
| `5c37d9a` | `run_shell` 批准提示，三種模式（auto/ask/safe）|
| `42d603a` | 子 agent 委派（`spawn_agent` 工具、深度限制）|
| `d7d28d6` | Lifecycle hooks（`Agent.on`、`--trace` demo flag）|
| `e3a9fa8` | Skills（`skills/<name>/SKILL.md` + `load_skill` 工具 + `/skills` 指令）|
| `0824aa0` | LLM provider 抽象（`--provider {anthropic,openai}` + Anthropic-canonical 訊息格式 + OpenAI 走 `/v1/responses` API + reasoning model 處理）|
| `abd1732` | Peer-to-peer multi-agent demo（`group_chat.py`：planner + coder + reviewer、N-1 滑動視窗廣播、`[DONE]` sentinel 終止、移除 `spawn_agent` + `remember` 防 group chat collapse、strict PLANNER_PROMPT 防 cosplay reviewer）|
| _(web 1)_ | Web UI commit 1：FastAPI + `POST /api/chat` + 原生 HTML/JS 單頁、非串流先看到完整回覆 |
| _(web 2)_ | Web UI commit 2：SSE 串流 + async LLM SDK 升級（新增 `text_chunk` lifecycle hook、`POST /api/chat` → request_id、`GET /api/stream` → EventSource、token-by-token 顯示;`Anthropic` / `OpenAI` SDK 換成 `AsyncAnthropic` / `AsyncOpenAI` 讓串流真正非阻塞）|
| _(web 3)_ | Web UI commit 3：工具呼叫卡片 + `run_shell` approval flow（`pre_tool` / `post_tool` hooks → tool_start / tool_end SSE 事件；`WebAgent` 覆寫 `_approve_run_shell` → SSE + Future + `POST /api/approve` 按鈕）|

照著讀的方式：`git checkout c1e9a04` 看最簡單的版本（~80 行），然後一路 `git log --oneline` 往新的 commit diff 過去。

---

## 適合誰讀

- 已經會 Python 基礎
- 想搞懂 LangChain / LlamaIndex / AutoGen 底下到底在做什麼
- 想自己寫 agent 但不想背一堆 framework API
- 想理解 Claude Code、Cursor、ChatGPT plugins 共通的核心架構

---

它是 **學習用的最小可用版本**：跑得起來、足以理解每個概念。
