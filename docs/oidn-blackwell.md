# Blackwell에서 ray-tracing denoiser 실패: 정책 관측 영상이 denoise되지 않음 (2026-10-16)

## 증상

잡 1685·1686(PRO 6000, `bundle-cu128-v5` 이미지)의 모든 정책 평가 worker 로그에 step마다 카메라 수만큼 다음 줄이 찍힌다.

```text
[svulkan2] [error] OIDN Error: invalid handle
```

bs1(RTX 2080 Ti, 같은 이미지)의 worker 로그에는 이 줄이 한 번도 없다.

## 원인

RoboTwin 2.0 `_base_task.setup_scene`은 무조건 ray tracing(`rt` shader, 32 spp, path depth 8)과 Intel Open Image Denoise(`oidn`) denoiser를 켠다. SAPIEN 3.0.0b1 wheel은 OIDN **2.0.1**을 `sapien/oidn_library/`에 동봉하고 `sapien/_oidn_tricks.py`가 이를 preload한다. OIDN 2.0.1은 NVIDIA Blackwell(sm_120) 이전 버전이라 CUDA device 생성이 실패하고, 이후 denoise 호출마다 `invalid handle`이 난다. 렌더링 자체는 계속되므로 **정책은 denoise되지 않은 32 spp path-tracing 프레임**을 관측한다. 공식 데이터 수집과 평가는 denoise된 영상을 쓰므로 이는 분포 이동이며 모든 모델의 성공률을 낮춘다. 상류 보고: [SAPIEN issue #243](https://github.com/haosulab/SAPIEN/issues/243) (RTX 5090, 사용자들이 OIDN 2.3.3 라이브러리 교체로 해결), [RoboTwin common issues](https://robotwin-platform.github.io/doc/common-issue/index.html). OIDN 2.3.3 release note: "Added NVIDIA Blackwell GPU support".

증거(같은 320×240 head camera, 첫 프레임의 Laplacian 표준편차): 클러스터 잡 1685 adjust_bottle 영상 22.6, bs1 2080 Ti 영상 12.1. [비교 이미지](../reports/oidn-blackwell-noise-compare.png) 왼쪽(클러스터)은 점 잡음이 보이고 오른쪽(2080 Ti)은 매끄럽다.

expert pool은 영상을 쓰지 않으므로 영향이 없다. 영향은 정책 평가 결과뿐이다.

## 수정 (v5 스냅샷)

이미지를 다시 만들지 않고 런처가 두 파일을 bind한다.

1. OIDN 2.3.3 Linux x86_64 배포본(`oidn-2.3.3.x86_64.linux.tar.gz`, sha256 `3c385230…796f`)을 `$DATA/downloads/oidn-2.3.3/`에 받아 검증·압축 해제하고 `lib/*.so*`를 `oidn_library/`로 모은 뒤 `…/site-packages/sapien/oidn_library`에 읽기 전용으로 bind한다.
2. ZIP의 `oidn/_oidn_tricks.py`(2.3.3 파일명을 preload)를 `…/sapien/_oidn_tricks.py`에 bind한다. `libsvulkan2.so`는 soname `libOpenImageDenoise.so.2`만 요구하므로 preload된 2.3.3이 그대로 쓰인다.

컨테이너 안에서 (a) 세 라이브러리와 loader의 checksum을 확인하고, (b) 실제로 rt+oidn 장면을 렌더링해 `/proc/self/maps`에 `libOpenImageDenoise_device_cuda.so.2.3.3`이 올라왔는지와 stderr에 `OIDN Error`가 없는지 검사한 뒤 `OIDN_PASS` 줄을 찍는다. 결과는 `reports/launchers/<jobid>/oidn-probe.json`에 남는다.

bs1(2080 Ti) 검증: 동봉 2.0.1과 교체 2.3.3 모두 오류 0, 같은 장면의 ground patch Laplacian 표준편차 0.28 vs 0.39(동일 수준), 2.3.3에서 `device_cuda.so.2.3.3` 로드 확인. Blackwell 효과는 클러스터 잡의 `OIDN_PASS`와 worker 로그에 오류 줄이 사라지는 것으로 확인한다.

## 영향과 처리

- 잡 1685(shift 1.0)·1686(denoise 없음)의 Fast-WAM 결과는 모두 폐기한다. v5 제출 전에 `runs/phys-pro6000-100-v6/fastwam/`을 `fastwam-oidn201-discarded/`로 옮긴다.
- pool·provenance·policy identity는 바뀌지 않는다. OIDN 버전은 fingerprint에 없으므로 런처가 `oidn-probe.json`으로 기록한다.
- v5는 에피소드 timeout도 1800초에서 3600초로 올린다(잡 1686은 place_bread_basket episode-0001이 30분을 넘겨 중단).
