# fix: bug fix

## Subject

Describe the **symptom** (what the user observed going wrong) not the
**fix** (what you changed in code). Reviewers see the diff — they
already know *how* you fixed it. They need to know *what was broken*.

Good (states the symptom):
- `fix(login): reject expired session tokens`
- `fix(billing): correct prorated refund for mid-cycle downgrades`

Avoid (states the diff):
- `fix(login): add token.exp check` — this is *how*, not *what*
- `fix: update billing.py` — file names aren't symptoms

Format: `fix(<scope>): <subject>`.

## Body

Three things, in order:

1. **Reproduction conditions** — what input or state triggers it
2. **Root cause** — why the code behaved wrong
3. **Impact range** — how long has this been broken, who's affected

The body is what makes a bug fix reviewable. Without it, a reviewer
can't tell whether your fix addresses the actual root cause or just
masks a symptom.

## Issue link

If there's a tracker issue, end the body with `Fixes #N` or
`Closes #N` — GitHub / GitLab will auto-close the issue when the
commit lands on the default branch.

## Example

```
fix(auth): reject expired session tokens

JWT tokens had a populated `exp` field but get_user never checked it,
so users with expired sessions continued to be authenticated until
they hit a route that re-validated the JWT independently. This
silently extended sessions beyond the configured TTL — affecting any
user idle for more than the 24h TTL since the v2.3 release.

Now get_user raises SessionExpired() when payload["exp"] < now,
which the auth middleware converts to a 401.

Fixes #847
```
