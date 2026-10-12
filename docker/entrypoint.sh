#!/usr/bin/env bash
set -euo pipefail
cd "$ROBOTWIN_PHYS_ROOT"
case "${1:-}" in
  doctor)
    shift
    exec python tools/doctor.py --output reports/docker-hardware.json "$@"
    ;;
  validate)
    shift
    exec python docker/validate_runtime.py "$@"
    ;;
  "") set -- robotwin-phys tasks ;;
esac
exec "$@"
