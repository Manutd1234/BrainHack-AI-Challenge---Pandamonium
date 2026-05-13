#!/usr/bin/env bash
# Archive duplicate Workbench clutter outside this repository.
#
# Usage:
#   bash tools/workbench_compact.sh          # dry run
#   bash tools/workbench_compact.sh --apply  # move duplicates into an archive
#
# This intentionally keeps the task folders inside the repository. The TIL CLI
# expects ae/, asr/, cv/, nlp/, and noise/ at repo root.

set -euo pipefail

APPLY=0
if [[ "${1:-}" == "--apply" ]]; then
  APPLY=1
fi

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
HOME_DIR="${HOME:-/home/jupyter}"
ARCHIVE_DIR="${HOME_DIR}/_pandamonium_archive_$(date +%Y%m%d_%H%M%S)"

real_repo="$(realpath -m "${REPO_ROOT}")"

is_inside_repo() {
  local path
  path="$(realpath -m "$1")"
  [[ "${path}" == "${real_repo}" || "${path}" == "${real_repo}/"* ]]
}

archive_path() {
  local source="$1"
  local name
  name="$(basename "${source}")"

  if is_inside_repo "${source}"; then
    echo "KEEP repo item: ${source}"
    return
  fi

  if [[ ! -e "${source}" ]]; then
    return
  fi

  if [[ "${APPLY}" == "1" ]]; then
    mkdir -p "${ARCHIVE_DIR}"
    echo "ARCHIVE: ${source} -> ${ARCHIVE_DIR}/${name}"
    mv "${source}" "${ARCHIVE_DIR}/${name}"
  else
    echo "WOULD ARCHIVE: ${source}"
  fi
}

echo "Repo root: ${REPO_ROOT}"
echo "Home dir:  ${HOME_DIR}"
echo ""

# Duplicate repo fragments commonly created by running commands from /home/jupyter.
for item in \
  ae asr cv nlp noise test til-26-curriculum \
  build_all.sh download_drive_data.py requirements-dev.txt simulate_evaluator.sh \
  submit_all.sh TERMINAL_GUIDE.sh SOLUTION.md; do
  archive_path "${HOME_DIR}/${item}"
done

# Old/alternate clones. The current clone is skipped automatically.
for item in BrainHack-AI-Challenge---Pandamonium pandamonium-repo; do
  archive_path "${HOME_DIR}/${item}"
done

echo ""
if [[ "${APPLY}" == "1" ]]; then
  echo "Done. Archived clutter under: ${ARCHIVE_DIR}"
else
  echo "Dry run only. Re-run with --apply to move these into an archive."
fi

echo ""
echo "Compact working layout:"
echo "  ${REPO_ROOT}/            # code, Dockerfiles, scripts"
echo "  ${HOME_DIR}/novice       # keep if this is TEAM_TRACK"
echo "  ${HOME_DIR}/advanced     # keep if this is TEAM_TRACK"
echo "  ${HOME_DIR}/pandamonium  # keep: til test result output"
