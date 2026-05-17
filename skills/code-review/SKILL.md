---
name: code-review
description: Review the current branch's changes against main for bugs and style
---

# Reviewing the current branch

You were asked to review code changes. Follow this procedure:

1. Find the base branch:
   - `run_shell` `git rev-parse --abbrev-ref HEAD` → current branch
   - `run_shell` `git rev-parse --verify main 2>/dev/null || git rev-parse --verify master`
2. Get the diff: `run_shell` `git diff <base>...HEAD`
3. For each changed file, identify:
   - **Bugs**: off-by-one, null/None handling, race conditions, resource leaks,
     wrong return types, missing error paths.
   - **Security**: injection (SQL, shell, path), auth bypass, secret leaks,
     unsafe deserialization, missing input validation at trust boundaries.
   - **Correctness**: does the change actually do what its commit/PR claims?
   - **Style**: only if it diverges from surrounding code in the same file.
4. Skip nitpicks the user can fix in a linter. Focus on things only a
   human reviewer would catch.

## Output format

Group findings by severity:

```
## Blocking
- <file:line> — <issue> — <suggested fix>

## Suggestions
- <file:line> — <issue>

## Nits
- <file:line> — <issue>
```

If you find no blocking issues, say so explicitly — don't pad with nits to
look thorough. A short clean review is more valuable than a long noisy one.
