# OC20 `*H` relaxation: 100계 LBFGS 후속 검증

2026-09-30 시작. 이 실험의 subset·모델·주요 판정 기준은 결과 확인 전에 고정했다.

## 질문과 고정 설정

앞선 [8계 PoC](relaxation_oc20_poc_2026-09-28.md)에서 선택한 **두 블록** 제안이, 아직 optimizer 비교에 쓰지 않은 다른 계에서도 같은 pretrained MLIP의 LBFGS relaxation 비용을 줄이는가?

- 데이터: 공식 OC20 `*H` per-adsorbate DFT 경로의 첫 1,000계. System ID hash로 나눈 train 823계와 validation 177계 중, 앞선 8계를 건너뛴 **validation 순번 9–108의 100계**를 평가한다. 이는 공식 OC20 ID/OOD validation이 아니며, 흡착종도 `*H` 하나다.
- 학생: 앞선 PoC의 seed 42, 4-head와 직접 4-step endpoint 모델. 재학습·checkpoint 선택을 하지 않는다. 둘 다 저장된 DFT 경로와 DFT force 입력으로 학습했지만, 이 평가의 제안 입력 force는 GemNet 예측값이다.
- MLIP: 동일한 공개 `GemNet-OC-S2EF-OC20-2M` direct-force checkpoint (`gemnet_oc_base_s2ef_2M.pt`, SHA-256 `a63092b7cf3c42231a5b60e8fb3a33000849e07e8b787698c94e3c1fc76abeca`). 세 방법의 시작 구조·제약·포텐셜은 같다.
- 기준: ASE LBFGS, `maxstep=0.2 Å`, `memory=100`, `alpha=70`, line search 미사용, `fmax < 0.05 eV/Å`, 최대 150 optimizer step. PDD와 직접 endpoint는 각각 최대 두 번 제안하고 GemNet energy가 0.1 eV 넘게 증가하거나 원자 최대 이동이 1 Å를 넘으면 거절한 뒤 같은 LBFGS로 보정한다.
- 비용: 제안 입력 force, 제안 후 guard, 보정 optimizer의 모든 GemNet 평가와 CPU wall time을 포함한다. 동일 모델 로딩 시간은 제외한다. 세 방법은 같은 프로세스에서 baseline, PDD, direct 순서로 실행되므로 wall time은 순서·캐시 영향도 함께 보고 호출 수와 교차 확인한다.

주요 비용 지표는 **100계 전체 GemNet 호출 수와 CPU wall time**, 수렴 계수, 계별 비용 차이 및 계 단위 paired bootstrap 95% 구간이다. 품질 진단은 baseline과 둘 다 수렴한 계에서 최종 GemNet energy가 baseline보다 `+0.05 eV` 또는 `+0.10 eV` 높은 횟수, DFT 최종 흡착 원자 위치 오차가 baseline보다 `+0.20 Å` 커진 횟수다. 이 기준은 다른 local minimum의 완전한 판별자가 아니므로 작은 오차가 나와도 같은 basin이라고 단정하지 않는다. 실패 계는 비용 합계에 포함한다. 100계를 본 뒤 threshold나 subset을 바꾸지 않는다.

```bash
MPLCONFIGDIR=data/oc20/cache/mpl XDG_CACHE_HOME=data/oc20/cache/xdg \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  .venv-oc20/bin/python scripts/relaxation_mlip_eval.py \
  --data data/oc20/H_poc_1000 \
  --checkpoint data/oc20/checkpoints/gemnet_oc_base_s2ef_2M.pt \
  --multi-head runs/relaxation/H_1000_L4_3000_force_seed42_multi_head.pt \
  --direct-endpoint runs/relaxation/H_1000_L4_3000_force_seed42_direct_endpoint.pt \
  --output runs/relaxation/gemnet_lbfgs_holdout100_two_blocks.json \
  --skip-systems 8 --systems 100 --max-steps 150 --fmax 0.05 \
  --guard-ev 0.1 --proposal-blocks 2 --proposal-force-source mlip --optimizer lbfgs
.venv/bin/python scripts/summarize_relaxation_eval.py \
  runs/relaxation/gemnet_lbfgs_holdout100_two_blocks.json
```

원시 JSON과 checkpoint는 Git에서 제외한다. 실행 중 매 method 결과를 원자적으로 JSON에 저장하며 `--resume`으로 같은 설정을 이어갈 수 있다.

## 결과

진행 중.
