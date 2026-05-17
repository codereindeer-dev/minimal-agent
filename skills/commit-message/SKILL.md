---
name: commit-message
description: Draft a Conventional Commits message from the current git diff
---

# Writing a commit message

1. `run_shell` `git status` to see staged / unstaged changes.
2. `run_shell` `git diff --staged` (or `git diff` if nothing is staged
   yet) to see the actual changes.
3. Decide the primary commit **type** by matching the diff against the
   table below.
4. `read_file` the matching type's recipe to get its subject / body
   conventions and an example.
5. Draft the message inside a fenced code block so the user can copy it.
6. **Do not run `git commit`** — the user decides when to commit.

## Types

| Type | Use for                                | Recipe          |
|------|----------------------------------------|-----------------|
| feat | Adding new user-facing functionality   | types/feat.md   |
| fix  | Fixing a bug (unintended behavior)     | types/fix.md    |

For other types (`refactor`, `docs`, `test`, `chore`, `perf`, etc.) no
recipe is bundled — apply standard Conventional Commits format. Subject
in imperative mood, body explains *why* not *what*.

## Multi-type diffs

If the diff genuinely spans multiple unrelated types (e.g. a feat plus
an unrelated docs change), suggest the user split the commit rather
than writing an umbrella message. State which files would go in each
split commit.
