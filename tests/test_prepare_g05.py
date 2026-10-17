"""prepare_g05 builds a bundle whose checkpoint resolves inside the bundle (job 1690 failure)."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).parents[1]
ORIGINAL = {
    'model': {'model_arch': {'hf_processor_path': 'checkpoints/qwen3_5_2b_base_processor'},
              'processor': {'tokenizer_params': {'pretrained_model_name_or_path': 'checkpoints/qwen3_5_2b_base_processor'}},
              'tokenizer': {'vq_config': {'ckpt_dir': 'checkpoints/action_tokenizer.pt'}}},
    'tokenizer': {'vq_config': {'ckpt_dir': 'checkpoints/action_tokenizer.pt'}},
    'data': {'action_size': 32},
}


def make_tree(tmp_path):
    tools = tmp_path / 'tools'
    tools.mkdir()
    shutil.copy(ROOT / 'tools/prepare_g05.py', tools / 'prepare_g05.py')
    source = tmp_path / 'checkpoints/g05'
    (source / 'g05-robotwin20/.hydra').mkdir(parents=True)
    (source / 'g05-robotwin20/checkpoints').mkdir()
    (source / 'qwen3_5_2b_base_processor').mkdir()
    (source / 'g05-robotwin20/.hydra/config.yaml').write_text(yaml.safe_dump(ORIGINAL))
    (source / 'g05-robotwin20/checkpoints/model_state_dict.pt').write_bytes(b'weights')
    (source / 'g05-robotwin20/dataset_stats.json').write_text('{}')
    (source / 'qwen3_5_2b_base_processor/tokenizer.json').write_text('{}')
    (source / 'action_tokenizer.pt').write_bytes(b'vq')
    return tools / 'prepare_g05.py', source, tmp_path / 'checkpoints/g05-runtime'


def run(script):
    return subprocess.run([sys.executable, str(script)], check=True, capture_output=True, text=True).stdout


def adapter_view(bundle):
    """What the pinned GalaxeaVLA adapter does: resolve the checkpoint, read the config beside it."""
    resolved = (bundle / 'checkpoints/model_state_dict.pt').resolve()
    return yaml.safe_load((resolved.parent.parent / '.hydra/config.yaml').read_text())


def test_bundle_checkpoint_resolves_inside_bundle_with_absolute_paths(tmp_path):
    script, source, bundle = make_tree(tmp_path)
    out = run(script)
    ckpt = bundle / 'checkpoints/model_state_dict.pt'
    assert not ckpt.is_symlink() and ckpt.resolve().parent.parent == bundle.resolve()
    assert os.path.samefile(ckpt, source / 'g05-robotwin20/checkpoints/model_state_dict.pt') and 'hard-linked' in out
    seen = adapter_view(bundle)
    processor = Path(seen['model']['model_arch']['hf_processor_path'])
    assert processor.is_absolute() and (processor / 'tokenizer.json').is_file()
    assert Path(seen['model']['processor']['tokenizer_params']['pretrained_model_name_or_path']) == processor
    assert Path(seen['tokenizer']['vq_config']['ckpt_dir']).is_file() and Path(seen['model']['tokenizer']['vq_config']['ckpt_dir']).is_file()
    # Originals untouched (their download lock checksums them).
    assert yaml.safe_load((source / 'g05-robotwin20/.hydra/config.yaml').read_text()) == ORIGINAL
    before = {p: p.stat().st_ino for p in bundle.rglob('*') if p.is_file()}
    run(script)  # idempotent
    assert before == {p: p.stat().st_ino for p in bundle.rglob('*') if p.is_file()}


def test_earlier_symlink_bundle_is_replaced(tmp_path):
    script, source, bundle = make_tree(tmp_path)
    (bundle / 'checkpoints').mkdir(parents=True)
    (bundle / 'checkpoints/model_state_dict.pt').symlink_to(source / 'g05-robotwin20/checkpoints/model_state_dict.pt')
    (bundle / 'dataset_stats.json').symlink_to(source / 'g05-robotwin20/dataset_stats.json')
    # With the symlink the adapter would read the ORIGINAL relative config: the job-1690 failure.
    (bundle / '.hydra').mkdir()
    (bundle / '.hydra/config.yaml').write_text('stale')
    assert adapter_view(bundle) == ORIGINAL
    run(script)
    assert not (bundle / 'checkpoints/model_state_dict.pt').is_symlink() and not (bundle / 'dataset_stats.json').is_symlink()
    assert Path(adapter_view(bundle)['model']['model_arch']['hf_processor_path']).is_absolute()


def test_unrelated_existing_files_and_missing_inputs_are_refused(tmp_path):
    script, source, bundle = make_tree(tmp_path)
    (bundle / 'checkpoints').mkdir(parents=True)
    (bundle / 'checkpoints/model_state_dict.pt').write_bytes(b'other weights!')
    with pytest.raises(subprocess.CalledProcessError) as err:
        run(script)
    assert 'unrelated' in err.value.stderr
    (bundle / 'checkpoints/model_state_dict.pt').unlink()
    (source / 'qwen3_5_2b_base_processor/tokenizer.json').unlink()
    with pytest.raises(subprocess.CalledProcessError) as err:
        run(script)
    assert 'tokenizer.json' in err.value.stderr
