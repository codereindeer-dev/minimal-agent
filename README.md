# minimal-agent

從零手寫的 AI agent loop，單檔 ~1000 行 Python。 Anthropic Claude SDK + native tools + MCP + 長期記憶 + 子 agent + lifecycle hooks。沒有 LangChain。

---

## 它是什麼

一個檔案（`minimal_agent.py`，約 1000 行）做完這些事：

- 跑 Claude 的 tool-use loop（request → tool call → tool result → 迴圈）
- 串流（streaming）輸出 assistant 訊息到 stdout
- 6 個原生工具：`run_shell` / `read_file` / `write_file` / `remember` / `recall` / `spawn_agent`
- 接 MCP server（`mcp-server-fetch`），把它的工具跟原生工具混在一起讓模型用
- 對話持久化（JSON 序列化、跨 session 接續）
- Context 管理：token 計算、自動 trim（rolling summary）、手動 `/compact`
- 長期記憶：Voyage AI embedding + JSONL append-only + cosine 相似度檢索
- 子 agent 委派：`spawn_agent` 工具、depth 限制 ≤ 2、context 隔離
- 6 個生命週期事件給外部 hook 註冊（`user_message`、`pre/post_turn`、`pre/post_tool`、`assistant_message`）

每個 git commit 是一個明確的小步進。 Clone 後 `git checkout` 早期 commit，從最簡單的版本一路讀上來。

---

## Quickstart

```bash
pip install anthropic python-dotenv mcp mcp-server-fetch voyageai
```

建立 `.env`：

```
ANTHROPIC_API_KEY=sk-ant-...
VOYAGE_API_KEY=...      # 只有 remember / recall 需要
```

跑：

```bash
python minimal_agent.py
```

```
you> 看一下現在目錄有什麼
claude> 我用 ls 看看。
  [native tool] run_shell({'command': 'ls'})
  [shell] Approve this command?
    > ls
    [y]es / [n]o / [a]lways (this session) > y
  [tool result] memory  minimal_agent.py  sessions  TUTORIAL.md ...
claude> 目錄裡有 5 個項目：...
[tokens: 2341]
```

---

## CLI 參數

| 參數 | 作用 |
|------|------|
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
| `/system` | 印出目前 system prompt |
| `/reset` | 清空對話歷史 |
| `/exit` | 離開 |

---

## 專案結構

```
minimal_agent.py    # 全部邏輯，單檔 ~1000 行
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

照著讀的方式：`git checkout c1e9a04` 看最簡單的版本（~80 行），然後一路 `git log --oneline` 往新的 commit diff 過去。

---

## 適合誰讀

- 已經會 Python 基礎
- 想搞懂 LangChain / LlamaIndex / AutoGen 底下到底在做什麼
- 想自己寫 agent 但不想背一堆 framework API
- 想理解 Claude Code、Cursor、ChatGPT plugins 共通的核心架構

---

它是 **學習用的最小可用版本**：跑得起來、足以理解每個概念。
