#!/usr/bin/env bash
# Campaign-root entry point for the CACE side of campaign_water_iface.
#
#   ./run_cace.sh [PYTHON] [extra run_cace_nnp.py flags...]
#
# Thin wrapper around cace/launch.sh so the pod's run_campaign.sh has one
# command per side. The deepmd side is run_deepmd.sh (owned by the deepmd
# harness). CACE provenance is recorded per run in runs/<arm>_s<X>/run_settings.json.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/cace/launch.sh" "$@"
