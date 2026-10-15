#!/usr/bin/env python3
"""Package the reviewed simulator snapshot and v5 OOD launchers."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAME = "robotwin-expert-fix-v1"


def main():
    source = ROOT / "src"
    files = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((source / "robotwin_phys").rglob("*.py"))}
    source_hash = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    archive = ROOT / "reports" / f"{NAME}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"name": NAME, "source_sha256": source_hash, "files": files,
                "robotwin_revision": "13c3c47ff4312dd62484bcd51be034af55c062d1"}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for relative in files:
            info = zipfile.ZipInfo(f"{NAME}/src/{relative}", date_time=(2026, 10, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, (source / relative).read_bytes())
        bundle.writestr(f"{NAME}/manifest.json", json.dumps(manifest, indent=2) + "\n")
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    host = '''
# Apply the reviewed simulator snapshot to the existing v5 image.
ROBOTWIN_PATCH_DIR="$DATA/patches/robotwin-expert-fix-v1"
ROBOTWIN_PATCH_ARCHIVE="$DATA/patches/robotwin-expert-fix-v1.zip"
mkdir -p "$DATA/patches"
(
  flock -x 9
  if [[ ! -d "$ROBOTWIN_PATCH_DIR/src/robotwin_phys" ]]; then
    if [[ ! -r "$ROBOTWIN_PATCH_ARCHIVE" ]]; then
      echo "Upload robotwin-expert-fix-v1.zip through OOD Files to $DATA/patches/" >&2
      exit 1
    fi
    printf '%s  %s\\n' ARCHIVE_HASH "$ROBOTWIN_PATCH_ARCHIVE" | sha256sum -c -
    ROBOTWIN_PATCH_STAGE=$(mktemp -d "$DATA/patches/.expert-fix-XXXXXX")
    trap 'rm -rf -- "$ROBOTWIN_PATCH_STAGE"' EXIT
    if command -v unzip > /dev/null; then
      unzip -q "$ROBOTWIN_PATCH_ARCHIVE" -d "$ROBOTWIN_PATCH_STAGE"
    elif command -v python3 > /dev/null; then
      python3 -m zipfile -e "$ROBOTWIN_PATCH_ARCHIVE" "$ROBOTWIN_PATCH_STAGE"
    else
      echo "Extract the ZIP through OOD Files into $DATA/patches/ before submitting." >&2
      exit 1
    fi
    mv "$ROBOTWIN_PATCH_STAGE/robotwin-expert-fix-v1" "$ROBOTWIN_PATCH_DIR"
  fi
) 9> "$DATA/patches/.expert-fix.lock"
export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_PATCH_DIR/src:$ROOT/src:ro"
'''.replace("ARCHIVE_HASH", archive_hash)
    payload = '''
# Check the entire source snapshot before creating or reading any pool.
"$ROOT/.env/bin/python" - <<'PY_EXPERT_PATCH'
import hashlib, json
from pathlib import Path
source = Path('/home/guest/robotwin-workspace/src')
files = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted((source / 'robotwin_phys').rglob('*.py'))}
actual = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
if actual != 'SOURCE_HASH':
    raise SystemExit('Expert patch source differs; restore the uploaded snapshot')
print('EXPERT_PLANNING_PATCH_PASS: ' + actual, flush=True)
PY_EXPERT_PATCH
'''.replace("SOURCE_HASH", source_hash)
    templates = [
        ("job-pro6000-single.sbatch", "job-pro6000-expert-fix.sbatch", "ood-fastwam-pro6000-v5-expert-fix.txt", False),
        ("job-pro6000-pi05.sbatch", "job-pro6000-pi05-expert-fix.sbatch", "ood-pi05-pro6000-v5-expert-fix.txt", False),
        ("job-pro6000-single.sbatch", "job-pro6000-expert-diagnose.sbatch", "ood-expert-diagnose-pro6000-v5.txt", True),
    ]
    for template, script, copy, diagnosis in templates:
        text = (ROOT / "cluster" / template).read_text()
        text = text.replace('"phys-pro6000-100-v1"', '"phys-pro6000-100-v2"')
        text = text.replace('export APPTAINERENV_VK_ICD_FILENAMES="$ROBOTWIN_NVIDIA_ICD"\n',
                            'export APPTAINERENV_VK_ICD_FILENAMES="$ROBOTWIN_NVIDIA_ICD"\n' + host)
        text = text.replace('ROOT=/home/guest/robotwin-workspace\nexport XDG_CACHE_HOME=',
                            'ROOT=/home/guest/robotwin-workspace\n' + payload + '\nexport XDG_CACHE_HOME=')
        if diagnosis:
            text = text.replace('#SBATCH --job-name=fastwam-phys', '#SBATCH --job-name=expert-diag')
            text = text.replace('#SBATCH --mem=96G', '#SBATCH --mem=32G')
            text = text.replace('#SBATCH --time=3-00:00:00', '#SBATCH --time=01:30:00')
            text = text.replace('exec bash "$ROOT/cluster/run_parallel.sh"',
                                'cd "$ROOT"\nexport SAPIEN_RENDER_DEVICE=cuda:0\n'
                                'exec "$ROOT/.env/bin/python" -m robotwin_phys.expert_diagnosis '
                                '--task adjust_bottle --seed 100054 '
                                '--output "$ROOT/reports/expert-diagnostics/$SLURM_JOB_ID"')
        (ROOT / "cluster" / script).write_text(text)
        (ROOT / "cluster" / copy).write_text(text)
    print(f"Packaged {archive}; simulator source {source_hash}")


if __name__ == "__main__":
    main()
