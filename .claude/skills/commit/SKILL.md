---
name: commit
description: "PROACTIVELY USE THIS SKILL when the user says \"commit\", \"land\", \"land this\", \"ship it\", \"done\", \"wrap up\", \"let's finish\", or any intent to commit work. This is the ONLY user entrypoint for the commit workflow — do not invoke /pre-land-review directly for commits. Orchestrates the full landing flow: pre-land review, staging, review marker, and git commit."
user-invocable: true
argument-hint: "[optional commit message override]"
---

# Commit — Full Landing Flow

You are landing work. Follow these steps exactly in order. Do not skip any
step. Do not create a git commit without completing the review.

## Step 1: Run /pre-land-review

Invoke the `pre-land-review` skill. This runs:
- All deterministic checks (ruff, check_projections.py)
- AI-powered sub-agent reviews (GPU code review, zero-copy enforcer,
  performance analysis, maintainability enforcer) as applicable

If the review finds BLOCKING issues, **stop here**. Fix them and re-run
`/commit`. Do not proceed to Step 2 with blocking findings.

## Step 2: Stage changes

After the review passes with verdict LAND:

1. Run `git status` to see what needs staging.
2. Run `git diff --cached --name-only` to see what is already staged.
3. Stage the appropriate files. Prefer staging specific files by name over
   `git add -A`. Never stage `.env`, credentials, or large binaries.
4. If the user specified which files to commit, stage only those.

## Step 3: Write the review marker

The review marker is a content-addressable record that the review passed. Write it:

```bash
printf '{\n  "timestamp": "%s",\n  "staged_hash": "%s",\n  "files": [%s],\n  "verdict": "LAND"\n}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$(git diff --cached | sha256sum | cut -d' ' -f1)" \
  "$(git diff --cached --name-only | sed 's/.*/"&"/' | paste -sd,)" \
  > .claude/.review-completed
```

**IMPORTANT**: If you stage any additional files AFTER writing the marker,
the hash will no longer match. Always write the marker as the LAST step
before `git commit`.

## Step 4: Create the commit

1. Analyze the staged diff to draft a concise commit message.
2. If the user provided `$ARGUMENTS`, use that as the commit message (or
   incorporate it).
3. Create the commit using a HEREDOC for proper formatting:

```bash
git commit -m "$(cat <<'EOF'
<commit message here>

Co-Authored-By: Claude <co-author tag>
EOF
)"
```

4. Run `git status` after the commit to verify success.

## Step 5: Report

Tell the user the commit was created. Show the commit hash and summary.

## Rules

- NEVER skip the pre-land review.
- NEVER use `--no-verify` or `-n` flags. The pre-land-gate hook blocks these.
- If the pre-commit hook fails (ruff, projections, etc.), fix the issues and
  retry. Do NOT amend — create a new commit.
- The review marker is single-use: clean it up after a successful commit.
- If the commit fails for any reason, re-run `/commit` from the top.

## Review Enforcement — Zero Tolerance

These rules govern how you handle findings from the pre-land review. They
are NON-NEGOTIABLE.

### NEVER reclassify BLOCKING as NIT

If a sub-agent flags something as BLOCKING, it stays BLOCKING. You do not
have authority to downgrade it. The only valid NIT is a known codebase-wide
gap that requires a coordinated migration — not "it works fine" or "it's
minor". When in doubt, it is BLOCKING.

### NEVER excuse new debt with existing debt

"Other functions don't do this either" is not a justification. Every new
line of code must meet the standard. If existing code has a shortcoming,
your new code must not compound that problem.

- If the fix is small (< 5 minutes): fix the upstream function too.
- If the fix is structural: spawn a `cuda-engineer` agent with the context
  needed to fix the upstream API, then build your code against the fixed
  version.

### NEVER accept CPU work in GPU code paths

If a review finds host round-trips (D->H->D), CPU fallbacks, or NumPy
calls inside a GPU-dispatched function, this is ALWAYS BLOCKING.

### What to do when a finding requires upstream fixes

Do NOT mark the review as LAND and leave a TODO. Instead:

1. Spawn a `cuda-engineer` agent (background, worktree) with:
   - The specific function/API that needs fixing
   - What it currently returns/does
   - What it should return/do
   - The downstream code that needs it
2. Wait for or incorporate the fix.
3. Re-run `/commit` against the complete change.
