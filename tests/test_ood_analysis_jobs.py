"""The pasted Job Composer scripts carry the current analysis scripts and paper table verbatim."""
import importlib.util
from pathlib import Path
import re

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('make_ood_jobs', ROOT / 'analysis/make_ood_jobs.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def heredoc(text, marker):
    return re.search(rf"<<'{marker}'\n(.*?)^{marker}\n", text, re.S | re.M).group(1)


def test_generated_jobs_match_committed_files_and_embed_current_scripts():
    export, analyze = m.export_job(), m.analyze_job()
    assert (ROOT / 'cluster/ood-export-release-v6.txt').read_text() == export
    assert (ROOT / 'cluster/ood-analyze-episodes-v6.txt').read_text() == analyze
    assert heredoc(export, 'ROBOTWIN_EXPORT') == (ROOT / 'analysis/export_release.py').read_text()
    assert heredoc(analyze, 'ROBOTWIN_ANALYZE') == (ROOT / 'analysis/analyze_episodes.py').read_text()
    assert heredoc(analyze, 'ROBOTWIN_TABLE') == (ROOT / 'references/paper-table4-physical-random.json').read_text()
    for text in (export, analyze):
        assert '--gres' not in text and 'set -euo pipefail' in text and 'DATA="$HOME/robotwin-data"' in text
    assert 'rm -rf "$OUT" "$OUT.zip"' in export and 'EXPORT_DONE' in export
    assert 'ANALYZE_DONE' in analyze and '--reference "$STAGE/paper-table4-physical-random.json"' in analyze
