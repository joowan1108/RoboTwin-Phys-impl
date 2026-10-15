import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('diagnose_provenance', ROOT / 'tools/diagnose_provenance.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_absent_helper_is_reported_with_actual_v2_source_path(tmp_path):
    source = tmp_path / 'manifests/phys-pro6000-100-v2/physical_random/adjust_bottle.json'
    write(source, {'status': 'verified', 'episodes': [{}, {}], 'rejected': [],
                   'provenance': {'files': {}, 'source_sha256': 'old-source'}})
    write(tmp_path / 'tools/expert_fix_previous.json', {'src/robotwin_phys/numerics.py': ['known']})
    write(tmp_path / 'reports/parallel/phys-pro6000-100-v4/fastwam-1641/provenance-0.json',
          {'files': {'src/robotwin_phys/numerics.py': 'known'}})
    before = source.read_bytes()
    result = d.report(tmp_path)
    assert result['selected_source'] == str(source)
    assert 'absent' in result['diagnosis']
    assert result['current_recorded_hashes']['src/robotwin_phys/numerics.py'] == 'known'
    assert source.read_bytes() == before


def test_unknown_hash_is_reported_without_allowing_it(tmp_path):
    source = tmp_path / 'manifests/phys-pro6000-100-v3/physical_random/adjust_bottle.progress.json'
    write(source, {'status': 'building', 'episodes': [], 'rejected': [],
                  'provenance': {'files': {'src/robotwin_phys/numerics.py': 'unknown'}}})
    write(tmp_path / 'tools/expert_fix_previous.json', {'src/robotwin_phys/numerics.py': ['known']})
    result = d.report(tmp_path)
    assert result['selected_source'] == str(source)
    assert 'unknown hash' in result['diagnosis']


def test_no_source_pool_is_reported(tmp_path):
    assert d.report(tmp_path)['selected_source'] is None
