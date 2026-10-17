# G0.5 정책 서버가 기동 직후 종료 (잡 1690)

## 증상

Fast-WAM 평가가 끝나고 `MODEL_START: g05` 직후 `Policy replica 0 exited`로 잡이 끝났다. `reports/parallel/phys-pro6000-100-v6/g05-1690/policy-0.log`:

```text
File ".../vendor/GalaxeaVLA/experiments/robotwin/galaxeafm_policy/deploy_policy.py", line 268, in _apply_checkpoint_model_config
FileNotFoundError: HF processor tokenizer.json not found: /opt/robotwin-policies/g05/workspace/vendor/GalaxeaVLA/checkpoints/qwen3_5_2b_base_processor
```

## 원인

고정 revision(89f2322)의 GalaxeaVLA RoboTwin 어댑터 `get_model`은 `ckpt_setting`을 `Path(...).resolve()`로 **symlink를 따라간 뒤** 그 파일 옆(`parent.parent/.hydra/config.yaml`)의 설정을 읽는다. 상류 README의 기본 배치는 저장소 안 `checkpoints/`에 HF bundle을 그대로 두는 것이고, 다운로드된 원본 `g05-robotwin20/.hydra/config.yaml`은 processor와 action tokenizer를 저장소 기준 상대 경로(`checkpoints/qwen3_5_2b_base_processor`, `checkpoints/action_tokenizer.pt`)로 가리킨다.

우리 `tools/prepare_g05.py`는 원본을 보존하고(다운로드 lock이 원본 checksum을 검증한다) 절대 경로를 넣은 설정을 `checkpoints/g05-runtime/.hydra/config.yaml`에 따로 만들었는데, 그 bundle의 `model_state_dict.pt`가 원본으로 가는 **symlink**였다. 어댑터가 symlink를 풀어 버리므로 우리 설정이 아니라 원본 설정을 읽었고, 상대 경로는 이미지 안 `vendor/GalaxeaVLA/checkpoints/`에 존재하지 않아 실패했다. bs1 서버(2080 Ti)에서는 bf16 정책을 띄울 수 없어 이 경로가 실제로 실행된 적이 없었다.

## 수정 (v6 ZIP, tools만 변경)

`tools/prepare_g05.py`가 bundle 파일을 symlink 대신 **hard link**(같은 파일시스템이 아니면 복사)로 만든다. `.resolve()`가 bundle 안에 머무르므로 어댑터는 절대 경로가 들어간 우리 설정을 읽는다. 이전에 만들어진 symlink bundle은 같은 원본을 가리키는 경우에만 교체하고, 원본 폴더는 건드리지 않는다. 준비 단계는 `prepare_cluster.py --models g05`가 모델 시작 전에 매번 실행하므로 ZIP만 올리면 된다. src는 바뀌지 않아 pool fingerprint와 Fast-WAM 결과에 영향이 없다.

`tests/test_prepare_g05.py`는 어댑터의 resolve 동작을 그대로 흉내 내어 symlink bundle이 원본 설정을 읽는 실패를 재현하고, 수정 후 bundle 설정(절대 경로, tokenizer.json 존재)을 읽는지, 재실행이 멱등인지, 무관한 파일을 덮어쓰지 않는지 확인한다.

## 적용 (v6 당시)

1. [robotwin-success-context-fix-v6.zip](../reports/robotwin-success-context-fix-v6.zip)을 `~/robotwin-data/patches/`에 올린다.
2. [ood-fastwam-pro6000x2-v6-context.txt](../cluster/ood-fastwam-pro6000x2-v6-context.txt)(v6 sha 고정)를 제출한다. 완료된 Fast-WAM은 그대로 두면 재검사 후 바로 `MODEL_DONE`으로 넘어간다. 로그에 `G05_RUNTIME_FIX` 줄과 prepare-model.log의 `checkpoint hard-linked` 줄이 있어야 한다.

## 두 번째 오류: `Missing key data` (잡 1693, v7 ZIP)

v6로 경로 문제를 넘긴 뒤 `get_model`이 `cfg.data.action_size`에서 `ConfigAttributeError: Missing key data`로 멈췄다. 상류 평가기 `experiments/robotwin/eval_robotwin_single.py`는 README 명령의 `task=robotwin`을 어댑터에 `sim_task`로 항상 넘기고, 어댑터는 이것을 Hydra override `task=robotwin`으로 조합한다. 이 task config가 `data: robotwin`(action_size 32), `model: g05`, `tokenizer: actioncodec`를 고른다. 우리 프로필은 이 값을 빠뜨렸고, 그러면 `sim_robotwin.yaml` → `train.yaml`의 빈 data group만 남는다.

수정: `configs/policies/g05.yaml`에 `args.sim_task: robotwin`. Fast-WAM 프로필처럼 v7 ZIP에 실어 평가 트리와 `/opt/robotwin-policies/g05/workspace/configs/policies/g05.yaml` 두 곳에 bind하고 컨테이너 안에서 checksum을 확인한다(`G05_SIM_TASK` 줄). G0.5 결과는 아직 없으므로 폐기할 것은 없고, Fast-WAM 프로필과 결과는 바뀌지 않는다. 나머지 상류 override(`action_horizon`·`sigma_shift` null, `num_inference_steps`는 모델 config의 10, bf16, replan 24)는 프로필에서 생략하거나 같은 값이라 상류와 같다. 상류는 `seed: 7`(train.yaml)을 넘기지만 우리 정책 서버는 episode마다 seed를 다시 정하므로 영향이 없다.

적용: [robotwin-success-context-fix-v7.zip](../reports/robotwin-success-context-fix-v7.zip)을 `~/robotwin-data/patches/`에 올리고 재생성된 런처를 제출한다. 로그에 `G05_RUNTIME_FIX`와 `G05_SIM_TASK`가 모두 있어야 한다.
