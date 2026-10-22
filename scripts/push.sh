# Commit training artifacts and push to GitHub.
# The token is read from $GIT_TOKEN (never written to disk in this repo).
set -uo pipefail
cd "$(dirname "$0")/.."

exec 9>/tmp/neurobrain_push.lock
flock 9

MSG=${1:-"train: checkpoint $(date -u +%FT%TZ)"}
REPO_SLUG=${REPO_SLUG:-dreamer2664/neurobrain}

if [ -z "${GIT_TOKEN:-}" ]; then
  echo "[push] GIT_TOKEN not set, skipping push"
  exit 0
fi

# Only artifacts. `git add -A` used to sweep whatever source edit happened to be
# in flight into a training commit (one half-resolved merge went out this way).
git add -A -- reports samples
[ -n "${EXTRA_PATHS:-}" ] && git add -A -- $EXTRA_PATHS
if git diff --cached --quiet; then
  echo "[push] nothing to commit"
else
  git -c user.name=neurobrain -c user.email=brain@neurobrain.local commit -q -m "$MSG"
fi

git remote remove origin 2>/dev/null || true
git remote add origin "https://x-access-token:${GIT_TOKEN}@github.com/${REPO_SLUG}.git"
git push -q origin main && echo "[push] pushed: $MSG" || echo "[push] push failed"
