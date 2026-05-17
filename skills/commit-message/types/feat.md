# feat: new functionality

## Subject

Imperative verb in the subject: `add`, `introduce`, `support`, `enable`.

Format: `feat(<scope>): <subject>` where `<scope>` is the affected
module/area (lowercase, optional but recommended).

Good:
- `feat(auth): add OAuth2 PKCE flow`
- `feat: support Markdown in user bio`

Avoid:
- `feat: Adding OAuth2` — wrong tense
- `feat: new feature for auth` — vague
- `feat(auth): added OAuth2 PKCE flow.` — past tense + trailing period

## Body

Explain **why** the feature was added — what user need or constraint
drove it, what trade-offs were considered, anything non-obvious for
the next reader. The *what* is in the diff.

Typical structure:

- The constraint / need that motivated this
- What this commit adds (one short sentence)
- Any feature flags, follow-ups, or things explicitly deferred

## Splitting

If the diff is > 200 lines or touches many subsystems, suggest
splitting into multiple feat commits — for example
`feat: skeleton` → `feat: wire X` → `feat: handle edge cases`.
Reviewers can step through one logical chunk at a time.

## Example

```
feat(auth): add OAuth2 PKCE flow

Mobile clients need an auth flow that doesn't require shipping a client
secret. PKCE solves this by deriving a per-request verifier on the
client and proving knowledge of it at the token endpoint.

This wires up /oauth/authorize and /oauth/token with PKCE verifier
support, gated behind the `feature_pkce` flag for staged rollout.
The existing client-credentials flow is untouched.
```
