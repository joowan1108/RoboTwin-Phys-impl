# 선택한 5개 RoboTwin checkpoint 연결

초기 평가 기반 구현에서 준비한 adapter와 checkpoint 정보입니다. 이후 추가한 Blackwell 모델별 Docker 이미지와 CSI Lab 실행은 [cluster.md](cluster.md)를 따릅니다. 아래 실제 학습 모델은 checkpoint load/inference 검증하지 않았습니다. 공식 adapter 코드와 checkpoint metadata를 확인해 연결 경로를 준비했으며, 현재 PC에서는 dummy policy를 통해 같은 RPC/observation/action 경로를 검증합니다. 이를 실제 모델 성공률로 보고하지 않습니다.

## 모델·artifact·환경 차이

| 모델 | 요청한 공식 HF repo와 사용할 파일 | 코드/추론 경로 |
|---|---|---|
| G0.5 | [OpenGalaxea/G05](https://huggingface.co/OpenGalaxea/G05), `g05-robotwin20/checkpoints/model_state_dict.pt` | [GalaxeaVLA](https://github.com/OpenGalaxea/GalaxeaVLA), official RoboTwin adapter |
| π0.5 | [motus-robotics/pi0.5_robotwin2](https://huggingface.co/motus-robotics/pi0.5_robotwin2), `model.safetensors` | RoboTwin pi05의 OpenPI runtime + model card의 training transform을 구현한 adapter |
| FACT | [Bariona/fact-wam](https://huggingface.co/Bariona/fact-wam), `transformer/` 및 `norm_stats_delta.json` | [FACT](https://github.com/Bariona/FACT) native inference server + official RoboTwin client adapter |
| Motus | [motus-robotics/Motus_robotwin2](https://huggingface.co/motus-robotics/Motus_robotwin2), `mp_rank_00_model_states.pt` | [Motus](https://github.com/thu-ml/Motus), `inference/robotwin/Motus/deploy_policy.py` |
| Fast-WAM | [yuanty/fastwam](https://huggingface.co/yuanty/fastwam), `robotwin_uncond_3cam_384.pt` | [FastWAM](https://github.com/yuantianyuan01/FastWAM), official RoboTwin adapter |

checkpoint와 code revision은 `configs/policies/*.yaml`에 고정되어 있습니다. HF API 응답은 `references/*-info.json`, 공개 model card는 `references/*-card.md`에 저장했습니다. G0.5는 gated repo여서 model card 원문 다운로드가 401로 거부되었으며, 공개 웹페이지/API metadata로 RoboTwin 파일 존재를 확인했습니다. 접근 권한을 사용자가 받은 뒤 다운로드해야 합니다.

- G0.5 공식 environment는 Python 3.10, Torch 2.7.1, Transformers 4.57.1 등이며 기본 SAPIEN 2.x dependency가 simulator와 충돌합니다.
- Fast-WAM 공식 environment는 Torch 2.7.1+cu128와 NumPy 2.2.6을 사용합니다. simulator의 NumPy 1.26/Torch 2.4.1에 함께 설치하지 않습니다.
- Motus 공식 RoboTwin guide는 CUDA ≥12.6과 추가 model dependency를 요구합니다. model card의 GPU 메모리 요구는 T5 pre-encoding 사용 시 약 24GB, 그렇지 않으면 약 41GB입니다.
- π0.5의 OpenPI runtime은 Python ≥3.11, JAX/Transformers/PyTorch의 별도 dependency를 사용합니다. 모델은 PyTorch checkpoint로 로드합니다. `adapt_to_pi=True`, delta joint actions, absolute grippers, horizon 32는 요청한 checkpoint card 기준입니다. stock RoboTwin pi05 config의 `adapt_to_pi=False`를 그대로 쓰지 않습니다.
- FACT native inference server는 별도 모델 환경이 필요합니다. 본 RPC bridge가 FACT client를 실행하고 그 client가 FACT server에 연결합니다. 이 경우 모델 random seed/identity의 최종 권한은 native server에 있으므로 native server 설정도 함께 보존해야 합니다.

RTX 2080 Ti에서는 이 문서의 공식 BF16 profile을 실행하지 않고 명확한 오류를 내도록 했습니다. 이는 모든 FP16/CPU-offload 변형이 불가능하다는 뜻은 아닙니다. 그런 변형은 별도 구현·성능 검증 후 다른 profile로 기록해야 합니다. 현재 제공하는 `minimum_vram_gb` 검사 외에 다른 모델의 VRAM 적합성을 실제 load 없이 보장하지 않습니다.

## 추후 다운로드

모델 repo를 simulator 환경에 설치하지 않고 소스만 가져옵니다.

```bash
cd /home/guest/robotwin-workspace
source tools/activate.sh
python tools/clone_policy_repos.py
python -m pip install huggingface-hub==0.34.4

# training optimizer와 LIBERO weights를 제외하고 RoboTwin weights/stats만 다운로드
python tools/download_models.py --models pi05 fact motus fastwam

# HF 웹에서 G05 조건에 동의하고 권한을 얻은 뒤 사용자의 인증을 설정
huggingface-cli login
python tools/download_models.py --models g05
python tools/prepare_g05.py
```

다운로드 도구는 SHA-256 검증과 `download-lock.json`을 남깁니다. 모델 실행을 추후로 정한 시점에 대용량 다운로드는 중단했으므로 현재 `checkpoints/`의 partial 파일을 완성된 모델로 사용하면 안 됩니다. 같은 명령으로 재개할 수 있습니다.

필수 부가 backbone도 모델별 공식 설명에 따라 준비합니다. Motus는 `checkpoints/backbones/Wan2.2-TI2V-5B`, `checkpoints/backbones/Qwen3-VL-2B-Instruct`를 사용하도록 설정했습니다. FACT는 `Wan-AI/Wan2.2-TI2V-5B-Diffusers`의 VAE/T5/tokenizer 등이 필요합니다. Fast-WAM 공식 runtime은 설정의 Wan2.2 및 Wan2.1 tokenizer asset을 추가로 로드합니다. task checkpoint 하나만으로 모든 backbone 다운로드가 끝났다고 가정하지 않습니다.

G0.5의 `prepare_g05.py`는 원래 download bundle을 보존하고 `g05-runtime/.hydra/config.yaml`에 로컬 processor/tokenizer 경로를 연결합니다. 모델 구조나 weight 값은 바꾸지 않습니다. gated config를 아직 실제로 읽을 수 없으므로 이 bundle 준비 단계는 권한을 얻은 뒤 검증해야 합니다.

## 모델 환경과 서버

각 공식 repository의 고정 revision 설치 안내를 따라 **서로 분리된 환경**을 만듭니다. 공통 bridge는 NumPy/PyYAML만 추가로 필요하고 SAPIEN/CuRobo를 import하지 않습니다. Python 3.11 환경에서는 이 프로젝트를 `pip install -e .`하지 말고 `PYTHONPATH`로 bridge를 노출합니다. 프로젝트의 simulator package는 Python 3.10으로 제한되어 있습니다.

```bash
export PYTHONPATH=/home/guest/robotwin-workspace/src
export CUDA_VISIBLE_DEVICES=0

# 원하는 모델 config 하나 선택. 한 서버가 순차적으로 여러 task를 처리 가능.
/path/to/fastwam-env/bin/python -m robotwin_phys.policy_server \
  --config /home/guest/robotwin-workspace/configs/policies/fastwam.yaml --port 9000

# 같은 명령 형식으로 g05.yaml, pi05.yaml, motus.yaml 사용
```

FACT는 먼저 native server를 시작합니다. model paths와 dependency 설치는 공식 [FACT checkpoint 안내](https://huggingface.co/Bariona/fact-wam)를 기준으로 합니다.

```bash
cd /home/guest/robotwin-workspace/vendor/FACT
/path/to/fact-env/bin/python -m scripts.inference_server \
  --model_id /path/to/Wan2.2-TI2V-5B-Diffusers \
  --transformer_path /home/guest/robotwin-workspace/checkpoints/fact/transformer \
  --stats_path /home/guest/robotwin-workspace/checkpoints/fact/norm_stats_delta.json \
  --port 8093

# 다른 터미널, 같은 FACT 환경
PYTHONPATH=/home/guest/robotwin-workspace/src \
  /path/to/fact-env/bin/python -m robotwin_phys.policy_server \
  --config /home/guest/robotwin-workspace/configs/policies/fact.yaml --port 9000
```

simulator는 어떤 모델이든 `robotwin-phys evaluate --manifest ... --policy-port 9000 --live --video`로 연결합니다. RPC는 upstream의 `get_model / eval / reset_model` interface를 유지하며, RGB·14차원 qpos·instruction을 전달하고 adapter가 원래대로 preprocessing/normalization/action chunk를 처리합니다. NumPy 배열은 길이 제한 JSON/base64로 전달합니다. pickle을 이용한 임의 객체 역직렬화를 사용하지 않습니다.

현재 profile의 replan 길이는 Fast-WAM 24, G0.5 24, Motus 공식 adapter, FACT 48, π0.5 32입니다. 이는 공개 평가 코드/체크포인트 설정에서 선택한 값이며 RoboTwin-Phys 저자의 미공개 상세 hyperparameter와 동일하다고 확인된 것은 아닙니다. 변경하면 model profile/config hash가 달라지며 새로운 run 폴더를 사용해야 합니다.

모든 모델에 대해 같은 **시뮬레이터 manifest**를 쓰는 것이 환경 조건의 공정성을 보장하는 핵심입니다. 모델 checkpoint별 추론 정밀도·normalization·chunk 길이는 각 모델 사양으로 따로 기록합니다.
