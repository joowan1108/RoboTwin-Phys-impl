#!/usr/bin/env python3
"""Embed the refill-aware reverification tool without changing the math ZIP."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    replacement = (ROOT / 'tools/reverify_pool.py').read_text()
    replacement_hash = hashlib.sha256(replacement.encode()).hexdigest()
    with zipfile.ZipFile(ROOT / 'reports/robotwin-expert-fix-v2.zip') as archive:
        manifest = json.loads(archive.read('robotwin-expert-fix-v2/manifest.json'))
    previous_hash = manifest['tools_sha256']
    tools = dict(manifest['tools'], **{'reverify_pool.py': replacement_hash})
    updated_hash = hashlib.sha256(json.dumps(tools, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    host = '''
# Only the reverification controller changes; simulator and v3 pool provenance
# remain identical to the uploaded math patch. No additional ZIP is required.
ROBOTWIN_REVERIFY_HOST="$DATA/reports/launchers/$SLURM_JOB_ID/reverify_pool.py"
mkdir -p "$(dirname -- "$ROBOTWIN_REVERIFY_HOST")"
cat > "$ROBOTWIN_REVERIFY_HOST" <<'ROBOTWIN_REFILL_TOOL'
''' + replacement + '''ROBOTWIN_REFILL_TOOL
printf '%s  %s\\n' REPLACEMENT_HASH "$ROBOTWIN_REVERIFY_HOST" | sha256sum -c -
export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_REVERIFY_HOST:$ROOT/tools/reverify_pool.py:ro"
'''.replace('REPLACEMENT_HASH', replacement_hash)
    templates = [
        ('job-pro6000-pool-geometry-v5.sbatch', 'job-pro6000-pool-refill-v5.sbatch', 'ood-pool-pro6000x2-v5-refill.txt'),
        ('job-pro6000-fastwam-geometry-v5.sbatch', 'job-pro6000-fastwam-refill-v5.sbatch', 'ood-fastwam-pro6000-v5-refill.txt'),
    ]
    for template, script, copy in templates:
        text = (ROOT / 'cluster' / template).read_text()
        text = text.replace('# Existing v5 images need the official template-generation step after unzip.',
                            host + '\n# Existing v5 images need the official template-generation step after unzip.')
        text = text.replace(previous_hash, updated_hash)
        (ROOT / 'cluster' / script).write_text(text)
        (ROOT / 'cluster' / copy).write_text(text)
    print('Generated refill-aware OOD scripts; no new math ZIP needed')


if __name__ == '__main__':
    main()
