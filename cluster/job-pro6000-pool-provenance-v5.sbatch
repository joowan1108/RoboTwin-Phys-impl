#!/bin/bash
#SBATCH --job-name=phys-pool-fixed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%j.out

# 사용자 설정: 이미지 하나가 시뮬레이터와 다섯 policy 환경을 포함합니다.
set -euo pipefail
IMAGE="docker://joowan1108/robotwin-phys:bundle-cu128-v5"
# Set HOME in the payload: Apptainer rejects APPTAINERENV_HOME.
# CMD is assigned below after generating the asset-preparation payload.
DATA="$HOME/robotwin-data"
export APPTAINERENV_ROBOTWIN_EXPERIMENT="phys-pro6000-100-v4"
export APPTAINERENV_ROBOTWIN_MODEL="fastwam"
export APPTAINERENV_ROBOTWIN_PHASE="freeze"
export APPTAINERENV_ROBOTWIN_EPISODES=100
export APPTAINERENV_ROBOTWIN_VIDEO_SELECTION="first"
# One allocated PRO 6000 runs one simulator and one FastWAM policy process.
# The evaluator uses SLURM's assigned GPU ID; no host GPU number is assumed.
export APPTAINERENV_ROBOTWIN_FREEZE_WORKERS_PER_GPU=4

# ---- (선택) private 이미지일 때만: 아래 두 줄 주석 해제 ----
# export APPTAINER_DOCKER_USERNAME="joowan1108"
# export APPTAINER_DOCKER_PASSWORD="$(cat ~/.dockerhub_token)"
# G05 접근 권한을 받은 HF token을 OOD Files에서 ~/.huggingface_token에 저장 후 주석 해제:
# export APPTAINERENV_HF_TOKEN="$(cat ~/.huggingface_token)"

# 사용자 영역의 실행 설정. 하단의 apptainer 명령은 그대로 사용합니다.
: "${SLURM_JOB_ID:?Submit through OOD Job Composer}"
: "${CUDA_VISIBLE_DEVICES:?SLURM must assign the GPUs}"
export APPTAINERENV_SLURM_JOB_ID="$SLURM_JOB_ID"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export APPTAINERENV_ROBOTWIN_DATA_ID="$DATA"
unset APPTAINERENV_HOME SINGULARITYENV_HOME
export APPTAINERENV_PYTHONUNBUFFERED=1
export PYTHONUNBUFFERED=1
export APPTAINER_CLEANENV=1
export APPTAINER_NO_HOME=1
export APPTAINER_NO_MOUNT="home,cwd"
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}/robotwin-apptainer-$SLURM_JOB_ID"
mkdir -p "$APPTAINER_TMPDIR" "$DATA/cache/home"
ROOT=/home/guest/robotwin-workspace
export APPTAINER_CWD="$ROOT"
# Use the image's glibc-compatible EGL/GLdispatch before host-injected copies.
# NVIDIA driver libraries remain available from /.singularity.d/libs.
export APPTAINERENV_LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/.singularity.d/libs"
export APPTAINER_BINDPATH="$DATA/cache/home:/tmp/robotwin-home"
for dir in assets checkpoints downloads manifests runs reports; do
  mkdir -p "$DATA/$dir"
  target="$ROOT/$dir"
  if [[ "$dir" = assets ]]; then target="$ROOT/vendor/RoboTwin/assets"; fi
  export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$DATA/$dir:$target"
done
for dir in /usr/share/vulkan/icd.d /etc/vulkan/icd.d /usr/share/glvnd/egl_vendor.d /etc/glvnd/egl_vendor.d; do
  if [[ -d "$dir" ]]; then export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$dir:$dir:ro"; fi
done
# Select NVIDIA only; unrelated host Mesa ICD files need not be loaded.
ROBOTWIN_NVIDIA_ICD=""
for icd in /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
  if [[ -r "$icd" ]]; then ROBOTWIN_NVIDIA_ICD="$icd"; break; fi
done
if [[ -z "$ROBOTWIN_NVIDIA_ICD" ]]; then
  echo "NVIDIA Vulkan ICD is missing on $(hostname)." >&2
  exit 1
fi
export APPTAINERENV_VK_ICD_FILENAMES="$ROBOTWIN_NVIDIA_ICD"

# Apply the reviewed simulator snapshot to the existing v5 image.
ROBOTWIN_PATCH_DIR="$DATA/patches/robotwin-curobo-fix-v1"
ROBOTWIN_PATCH_ARCHIVE="$DATA/patches/robotwin-curobo-fix-v1.zip"
mkdir -p "$DATA/patches"
(
  flock -x 9
  if [[ ! -d "$ROBOTWIN_PATCH_DIR/src/robotwin_phys" ]]; then
    if [[ ! -r "$ROBOTWIN_PATCH_ARCHIVE" ]]; then
      echo "Upload robotwin-curobo-fix-v1.zip through OOD Files to $DATA/patches/" >&2
      exit 1
    fi
    printf '%s  %s\n' f7c7deb0045f0459ad9daa74d080c0a37f991fb841b0fdd436fddaaa51003454 "$ROBOTWIN_PATCH_ARCHIVE" | sha256sum -c -
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
    mv "$ROBOTWIN_PATCH_STAGE/robotwin-curobo-fix-v1" "$ROBOTWIN_PATCH_DIR"
  fi
) 9> "$DATA/patches/.expert-fix.lock"
export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_PATCH_DIR/src:$ROOT/src:ro"
export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_PATCH_DIR/tools:$ROOT/tools:ro"
export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_PATCH_DIR/vendor/transforms.py:$ROOT/vendor/RoboTwin/envs/utils/transforms.py:ro"
export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_PATCH_DIR/vendor/planner.py:$ROOT/vendor/RoboTwin/envs/robot/planner.py:ro"


# Correct the reviewed-helper addition check without changing simulator source.
ROBOTWIN_REVERIFY_HOST="$DATA/reports/launchers/$SLURM_JOB_ID/reverify_pool.py"
mkdir -p "$(dirname -- "$ROBOTWIN_REVERIFY_HOST")"
cat > "$ROBOTWIN_REVERIFY_HOST" <<'ROBOTWIN_PROVENANCE_CONTROLLER'
#!/usr/bin/env python3
"""Reverify v2 expert episodes before continuing a pool with the math repair."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from robotwin_phys.cli import worker
from robotwin_phys.conditions import digest
from robotwin_phys.manifest import atomic_json, seal
from robotwin_phys.simulator import assert_same_state

REVIEWED_SOURCE_CHANGES = {'src/robotwin_phys/numerics.py', 'src/robotwin_phys/planning.py',
                           'src/robotwin_phys/cli.py', 'src/robotwin_phys/curobo_compat.py',
                           'src/robotwin_phys/simulator.py', 'src/robotwin_phys/provenance.py'}
REVIEWED_SOURCE_ADDITIONS = {'src/robotwin_phys/numerics.py', 'src/robotwin_phys/curobo_compat.py'}


def approved_upgrade(previous, current, approved):
    if previous.get('source_sha256') != digest(previous['files']):
        raise ValueError('Previous source provenance hash is invalid')
    for key in previous.keys() | current.keys():
        if key not in ('source_sha256', 'files') and previous.get(key) != current.get(key):
            if key == 'curobo' and previous.get(key) is None:
                from robotwin_phys.curobo_compat import PIN, SOURCE_SHA256, FIX
                if current.get(key) == dict(revision=PIN, motion_gen_sha256=SOURCE_SHA256, compatibility_fix=FIX):
                    continue
            raise ValueError(f'Reverification requires unchanged runtime/config/assets: {key}')
    def reviewed(path, old):
        value = approved.get(path)
        if path not in previous['files'] and path in REVIEWED_SOURCE_ADDITIONS:
            # Known v3 helper hashes also describe a reviewed addition to v2.
            return value is None or current['files'].get(path) in (value if isinstance(value, list) else [value])
        return old in value if isinstance(value, list) else old == value
    for path in previous['files'].keys() | current['files'].keys():
        old, new = previous['files'].get(path), current['files'].get(path)
        if old == new:
            continue
        if path.startswith('src/robotwin_phys/'):
            if path not in REVIEWED_SOURCE_CHANGES or not reviewed(path, old):
                raise ValueError(f'Unreviewed previous simulator source: {path}; '
                                 f'previous_sha256={old}; current_sha256={new}; approved={approved.get(path)}')
        elif path in ('vendor/RoboTwin/envs/utils/transforms.py', 'vendor/RoboTwin/envs/robot/planner.py'):
            if not reviewed(path, old):
                raise ValueError('Unreviewed previous transform implementation')
        else:
            raise ValueError(f'Unreviewed benchmark change: {path}')


def reverify(source, target, current, approved, timeout=1800):
    source, target = Path(source), Path(target)
    if target.exists() or target.with_suffix('.progress.json').exists():
        return  # Already migrated; normal freeze validates and resumes this pool.
    source = source if source.exists() else source.with_suffix('.progress.json')
    prior = json.loads(source.read_text())
    if prior.get('sha256') != seal(prior)['sha256'] or prior.get('schema') != 'robotwin-phys-local-v1':
        raise ValueError('Source manifest/progress failed integrity validation')
    if prior.get('status') not in ('verified', 'building'):
        raise ValueError('Only verified or expert-building source pools can be reverified')
    approved_upgrade(prior['provenance'], current, approved)
    source_hash = prior['sha256']
    folder = target.parent / (target.stem + '-reverification')
    progress = target.with_suffix('.reverify.progress.json')
    audit = dict(source=str(source), source_sha256=source_hash, current_provenance_sha256=digest(current),
                 episodes=[], decisions=[], rejected=[], status='reverifying')
    if progress.exists():
        saved = json.loads(progress.read_text())
        if saved.get('sha256') != seal(saved)['sha256']:
            raise ValueError('Reverification progress was modified')
        if saved['source_sha256'] != source_hash or saved['current_provenance_sha256'] != digest(current):
            raise ValueError('Reverification source or runtime changed')
        audit = {k: v for k, v in saved.items() if k != 'sha256'}
    old_episodes = prior['episodes']
    if len(audit['episodes']) > len(old_episodes):
        raise ValueError('Invalid reverification episode count')
    # The previous tool stopped on the first rejected candidate. Its saved
    # episodes are therefore a verified prefix of the source episode list.
    if 'decisions' not in audit:
        audit['decisions'] = [dict(source_episode_index=index, seed=episode['seed'],
                                  condition_sha256=episode['condition_sha256'], status='verified')
                              for index, episode in enumerate(audit['episodes'])]
        audit['rejected'] = []
    decisions = audit['decisions']
    if len(decisions) > len(old_episodes):
        raise ValueError('Invalid reverification decision count')
    retained = iter(audit['episodes'])
    for index, decision in enumerate(decisions):
        original = old_episodes[index]
        if (decision['source_episode_index'] != index or decision['seed'] != original['seed'] or
                decision['condition_sha256'] != original['condition_sha256']):
            raise ValueError('Reverification decision does not match saved seed conditions')
        if decision['status'] == 'verified':
            episode = next(retained, None)
            if (episode is None or episode['seed'] != original['seed'] or
                    episode['condition_sha256'] != original['condition_sha256']):
                raise ValueError('Reverification progress does not match saved seed conditions')
        elif decision['status'] != 'rejected':
            raise ValueError('Invalid reverification decision status')
    if next(retained, None) is not None:
        raise ValueError('Reverification episodes have no matching decisions')
    for index, episode in enumerate(audit['episodes']):
        if (episode['episode_index'] != index or digest(episode['condition']) != episode['condition_sha256'] or
                digest(episode['initial_state']) != episode['initial_state_sha256'] or
                episode.get('expert_success') is not True):
            raise ValueError('Invalid retained reverification episode')
    rejected_decisions = [decision for decision in decisions if decision['status'] == 'rejected']
    if len(rejected_decisions) != len(audit['rejected']):
        raise ValueError('Reverification rejection history has no matching decisions')
    for decision, rejected in zip(rejected_decisions, audit['rejected']):
        if (rejected['seed'] != decision['seed'] or rejected['reason'] != decision['reason'] or
                rejected.get('reason') not in ('expert_failed', 'unstable_initialization')):
            raise ValueError('Invalid reverification rejection history')
    for index, episode in enumerate(old_episodes):
        if (episode['episode_index'] != index or episode['seed'] != episode['condition']['episode_seed'] or
                digest(episode['condition']) != episode['condition_sha256'] or
                digest(episode['initial_state']) != episode['initial_state_sha256'] or
                episode.get('expert_success') is not True):
            raise ValueError('Invalid saved expert episode')
        if index < len(decisions):
            continue
        result = worker(dict(mode='expert', episode=episode, instruction_type=prior['instruction_type']),
                        folder, f'seed-{episode["seed"]}', timeout)
        decision = dict(source_episode_index=index, seed=episode['seed'],
                        condition_sha256=episode['condition_sha256'])
        if result.get('status') == 'rejected' and result.get('reason') in ('expert_failed', 'unstable_initialization'):
            audit['rejected'].append(dict(result, seed=episode['seed'],
                                          condition_sha256=episode['condition_sha256'],
                                          reverification=True, prior_expert_success=True))
            decisions.append(dict(decision, status='rejected', reason=result['reason']))
            atomic_json(progress, seal(audit))
            print(f'Reverification rejected seed {episode["seed"]}: {result["reason"]}; '
                  'the missing episode will be filled with a new expert-verified seed', flush=True)
            continue
        if result.get('status') != 'verified':
            raise RuntimeError(f'Saved expert seed {episode["seed"]} no longer verifies: {result}; '
                               'runtime errors are not excluded as infeasible')
        verified = result['episode']
        assert_same_state(episode['initial_state'], verified['initial_state'])
        if verified['instruction'] != episode['instruction']:
            raise ValueError('Reverification changed the saved instruction')
        verified = dict(verified, episode_index=len(audit['episodes']))
        audit['episodes'].append(verified)
        decisions.append(dict(decision, status='verified'))
        atomic_json(progress, seal(audit))
        print(f'Reverified {prior["task"]} {index+1}/{len(old_episodes)}: seed {episode["seed"]}', flush=True)
    migrated = {k: v for k, v in prior.items() if k != 'sha256'}
    migrated.update(provenance=current, episodes=audit['episodes'], rejected=prior['rejected'] + audit['rejected'])
    migrated['status'] = 'verified' if len(migrated['episodes']) == migrated['requested_episodes'] else 'building'
    atomic_json(target.with_suffix('.progress.json'), seal(migrated))
    if migrated['status'] == 'verified':
        atomic_json(target, seal(migrated))
    audit['status'] = 'reverified'
    audit['selection'] = 'Retain reverified prior seeds; reject infeasible seeds and refill from the next unused candidate'
    atomic_json(target.with_suffix('.reverification.json'), seal(audit))
    print(f'REVERIFICATION_PASS {prior["task"]}: {len(audit["episodes"])}/{len(old_episodes)} saved episodes retained; '
          f'{len(audit["rejected"])} rejected; {migrated["requested_episodes"] - len(audit["episodes"])} '
          'episodes still needed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    from robotwin_phys.provenance import fingerprint
    approved = json.loads((ROOT / 'tools/expert_fix_previous.json').read_text())
    reverify(args.source, args.target, fingerprint(), approved, args.timeout)


if __name__ == '__main__':
    main()
ROBOTWIN_PROVENANCE_CONTROLLER
printf '%s  %s\n' e4e26c2d48ee39c37517628b44b516e1cc49ce55e00db7138bb1c1a8e1bcf557 "$ROBOTWIN_REVERIFY_HOST" | sha256sum -c -
export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_REVERIFY_HOST:$ROOT/tools/reverify_pool.py:ro"

# Existing v5 images need the official template-generation step after unzip.
# Write the payload through the reports bind; no image rebuild is required.
ROBOTWIN_LAUNCH_HOST="$DATA/reports/launchers/$SLURM_JOB_ID"
ROBOTWIN_LAUNCH_CONTAINER="$ROOT/reports/launchers/$SLURM_JOB_ID"
mkdir -p "$ROBOTWIN_LAUNCH_HOST"
cat > "$ROBOTWIN_LAUNCH_HOST/run.sh" <<'ROBOTWIN_LAUNCH'
#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/guest/robotwin-workspace

# Check the entire source snapshot before creating or reading any pool.
"$ROOT/.env/bin/python" - <<'PY_EXPERT_PATCH'
import hashlib, json
from pathlib import Path
root = Path('/home/guest/robotwin-workspace')
def hashes(folder, patterns):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob('*')) if p.is_file() and p.suffix in patterns
            and '__pycache__' not in p.parts}
def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
if checksum(hashes(root / 'src', ('.py',))) != '6dfbef9791d525c4d9261a595f87efebf9955c4a0d976ec382024816300fe5e9':
    raise SystemExit('Expert patch source differs; restore uploaded snapshot')
if checksum(hashes(root / 'tools', ('.py', '.sh', '.json'))) != '7b10dabe8f52ccf85348b2e50af7830f234c7165148d2abe2d8bef0fdb659007':
    raise SystemExit('Expert patch tools differ; restore uploaded snapshot')
if hashlib.sha256((root / 'vendor/RoboTwin/envs/utils/transforms.py').read_bytes()).hexdigest() != '942ed2b255094190f7380df113b78daed687803732928d23f2d9823d18862c53':
    raise SystemExit('Shared transform repair was not bound correctly')
if hashlib.sha256((root / 'vendor/RoboTwin/envs/robot/planner.py').read_bytes()).hexdigest() != '61e3d0d4fa53e58a2289a533a5bcf4d85a067bbd1939745a87bc19a7c6884f56':
    raise SystemExit('Planner exception repair was not bound correctly')
print('CUROBO_MASK_PATCH_SOURCE_PASS: simulator and reviewed helpers verified', flush=True)
PY_EXPERT_PATCH

export XDG_CACHE_HOME="$HOME/.cache" XDG_CONFIG_HOME="$HOME/.config"
export HF_HOME="$XDG_CACHE_HOME/huggingface" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$TORCH_EXTENSIONS_DIR"
# Use the same preparation lock as parallel_eval; assets are shared across jobs.
(
  flock -x 9
  cd "$ROOT"
  "$ROOT/.env/bin/python" tools/download_assets.py
  cd "$ROOT/vendor/RoboTwin"
  "$ROOT/.env/bin/python" script/update_embodiment_config_path.py < /dev/null
  for arm in left right; do
    target="assets/embodiments/aloha-agilex/curobo_${arm}.yml"
    if [[ ! -s "$target" ]]; then
      echo "Planner configuration was not generated: $target" >&2
      exit 1
    fi
  done
  echo "ASSET_PREPARATION_PASS: aloha-agilex left/right planner configs generated"
) 9> "$ROOT/downloads/.prepare.lock"
exec bash "$ROOT/cluster/run_parallel.sh" --freeze-workers-per-gpu "$ROBOTWIN_FREEZE_WORKERS_PER_GPU" --resume-from-experiment phys-pro6000-100-v3 --fallback-experiment phys-pro6000-100-v2
ROBOTWIN_LAUNCH
CMD="env HOME=/tmp/robotwin-home bash $ROBOTWIN_LAUNCH_CONTAINER/run.sh"

# ---- 캐시를 노드 로컬로 (NAS 홈 부담 방지, 건드리지 마세요) ----
export APPTAINER_CACHEDIR="/tmp/apptainer-$(whoami)"
mkdir -p "$APPTAINER_CACHEDIR"

# ============================================================
#  아래는 수정할 필요 없습니다.
# ============================================================

echo "=== Job start: $(date) on $(hostname) ==="
nvidia-smi                        # 할당된 GPU 확인

# 컨테이너 실행 (--nv = GPU 전달). 할당된 GPU만 보입니다.
apptainer exec --nv "$IMAGE" $CMD

echo "=== Job end: $(date) ==="
