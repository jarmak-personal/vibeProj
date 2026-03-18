#!/usr/bin/env bash
# Compare performance of the current working tree against a baseline.
#
# Strategy:
#   1. Run benchmarks on the CURRENT state (staged + unstaged changes).
#   2. Determine the baseline:
#      - If on a PR branch: merge-base of main and HEAD.
#      - If on main: HEAD (compares uncommitted changes vs last commit).
#   3. Create a temporary worktree at the baseline commit.
#   4. Install baseline vibeproj into the CURRENT venv (swap the code only).
#   5. Run benchmarks (same venv, same CuPy — only vibeproj code differs).
#   6. Restore current vibeproj, compare results.
#
# This ensures both runs share the identical environment (including CuPy/GPU).
#
# Usage:
#   ./benchmarks/bench_compare.sh [--threshold PCT] [--n COORDS]
#
# Examples:
#   ./benchmarks/bench_compare.sh                    # default: 15% threshold, 1M coords
#   ./benchmarks/bench_compare.sh --threshold 10     # stricter
#   ./benchmarks/bench_compare.sh --n 500000         # fewer coords (faster)

set -euo pipefail

THRESHOLD=15.0
N_COORDS=1000000
MAIN_BRANCH="main"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --threshold) THRESHOLD="$2"; shift 2 ;;
        --n)         N_COORDS="$2"; shift 2 ;;
        *)           echo "Unknown arg: $1"; exit 1 ;;
    esac
done

REPO_ROOT=$(git rev-parse --show-toplevel)
BENCH_SCRIPT="$REPO_ROOT/benchmarks/bench_projections.py"
TMPDIR=$(mktemp -d)
CURRENT_JSON="$TMPDIR/current.json"
BASE_JSON="$TMPDIR/base.json"

cleanup() {
    # Restore current vibeproj in the venv
    echo "Restoring current vibeproj..."
    cd "$REPO_ROOT"
    uv pip install -e . --quiet 2>/dev/null || true
    # Remove worktree if it exists
    if [ -n "${WORKTREE_DIR:-}" ] && [ -d "$WORKTREE_DIR" ]; then
        git worktree remove --force "$WORKTREE_DIR" 2>/dev/null || true
    fi
    rm -rf "$TMPDIR"
}
trap cleanup EXIT

cd "$REPO_ROOT"

# ── Step 1: Determine baseline commit ─────────────────────────────────
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

if [ "$CURRENT_BRANCH" = "$MAIN_BRANCH" ]; then
    # On main: compare against HEAD (working tree changes vs last commit)
    BASE_REF="HEAD"
    echo "On $MAIN_BRANCH — comparing HEAD vs working tree"
else
    # On a feature branch: compare against merge-base with main
    BASE_REF=$(git merge-base "$MAIN_BRANCH" HEAD)
    SHORT_BASE=$(git rev-parse --short "$BASE_REF")
    echo "On branch '$CURRENT_BRANCH' — comparing merge-base ($SHORT_BASE) vs current"
fi

# ── Step 2: Benchmark CURRENT state ───────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo " Benchmarking CURRENT state"
echo "═══════════════════════════════════════════════════════"
uv run "$BENCH_SCRIPT" run --n "$N_COORDS" --output "$CURRENT_JSON"

# ── Step 3: Swap to baseline code and benchmark ───────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo " Benchmarking BASELINE ($BASE_REF)"
echo "═══════════════════════════════════════════════════════"

WORKTREE_DIR="$TMPDIR/worktree"
git worktree add --detach "$WORKTREE_DIR" "$BASE_REF" --quiet

# Install baseline vibeproj into the CURRENT venv (swaps only the library code).
# This keeps CuPy, numpy, pyproj etc. identical between both runs.
echo "Installing baseline vibeproj into current venv..."
uv pip install -e "$WORKTREE_DIR" --quiet

# Run benchmark using the current bench script (it may not exist in the baseline)
# but with the baseline vibeproj code installed in the venv.
cd "$REPO_ROOT"
uv run "$BENCH_SCRIPT" run --n "$N_COORDS" --output "$BASE_JSON"

# ── Step 4: Restore current code and compare ──────────────────────────
echo ""
echo "Restoring current vibeproj..."
uv pip install -e "$REPO_ROOT" --quiet

echo ""
uv run "$BENCH_SCRIPT" compare "$BASE_JSON" "$CURRENT_JSON" --threshold "$THRESHOLD"
