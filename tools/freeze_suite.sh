#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/activate.sh"
cd "$ROBOTWIN_PHYS_ROOT"
protocol="${1:-physical_random}"
while IFS= read -r task; do
  target="manifests/$protocol/$task.json"
  if [[ -f "$target" ]]; then
    python -c 'from robotwin_phys.manifest import load; import sys; load(sys.argv[1])' "$target"
    continue
  fi
  python -m robotwin_phys.cli freeze --task "$task" --protocol "$protocol" --episodes 100 --seed 100000 --resume --output "$target"
done < <(python -m robotwin_phys.cli tasks)
