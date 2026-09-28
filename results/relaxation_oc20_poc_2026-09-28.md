# OC20 DFT 경로를 이용한 relaxation 블록 예측 PoC

2026-09-28. **결과 범위:** 실제 OC20 DFT relaxation 궤적에서 4개 내부 구조를 한 번의 모델 평가로 예측하는 오프라인 실험과, 공개 사전학습 GemNet-OC로 블록 제안 후 LBFGS refinement를 시험한 소규모 CPU 진단이다. 앞선 FIRE 결과도 참고용으로 남긴다. GPU 규모의 relaxation 가속 결과는 아니다.

## 데이터와 재현

- [OC20 공식 `*H` per-adsorbate 궤적](https://facebookresearch.github.io/fairchem/oc20/)의 `1.tar`(890,265,852 byte). 공식 MD5 `3697f04faf04251a23da8b88a78209f7` 일치.
- archive 순서의 첫 1,000개 서로 다른 계를 선택했다. System ID의 SHA-256 hash로 train 823개 / validation 177개를 분리했다. 같은 계의 프레임이 양쪽에 들어가지 않는다. 공식 OC20 ID/OOD validation은 아직 사용하지 않았다.
- 각 경로의 처음 6개 시작 프레임에서 연속 4-step target을 만든다. Train 4,938개 / validation 1,062개 윈도우. 고정 원자는 loss와 좌표 갱신에서 제외하고, PBC 최소 이미지 이동을 사용한다. 경로 길이 중앙값 44 frame, p90 124 frame.
- Mac arm64 CPU, PyTorch 2.13.0. 이 로컬 환경의 CUDA 및 MPS는 비활성이다. CPU 학습 시간은 GPU에서의 속도 증거가 아니다.
- 모델은 작은 방향 보존 pair-message backbone(width 32)이다. 동일 shared backbone에 4개의 incremental displacement head를 둔 모델과 1개의 endpoint head 모델을 비교했다. 양쪽 모두 시작 프레임에 **저장된 DFT force**를 입력으로 받는다. 이는 실제 MLIP force를 사용한 시험이 아니라, 사용자가 제안한 DFT≈MLIP 가정의 *oracle 입력* 검사다.
- Seed 42/43/44, 모델마다 Adam 3,000 update, learning rate 1e-3, 초기 6-window 균일 샘플링, adsorbate loss 가중치 4배. 숫자는 validation의 움직이는 원자별 4-step 위치 오차의 평균(Å)이다.

```bash
curl -L --fail 'https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/1.tar' -o data/oc20/H_1.tar
python scripts/relaxation_poc.py prepare --tar data/oc20/H_1.tar --output data/oc20/H_poc_1000 --max-systems 1000 --block 4
for seed in 42 43 44; do
  python scripts/relaxation_poc.py train --data data/oc20/H_poc_1000 \
    --output runs/relaxation/H_1000_L4_3000_force_seed${seed}.json \
    --block 4 --updates 3000 --windows-per-system 6 --start-mode early \
    --width 32 --threads 4 --seed ${seed} --use-forces
done
```

Data와 checkpoint, 원시 JSON은 Git에서 제외한다. `scripts/relaxation_poc.py`는 공식 tar의 `.extxyz.xz` 멤버를 스트리밍해 `npz`로 캐시한다. Tar 멤버의 path를 파일시스템에 추출하지 않는다.

## 측정값

| 방식 | 4-step 가동 원자 오차 (Å) | adsorbate 원자 오차 (Å) |
|---|---:|---:|
| 이동 없음 | 0.09880 | 0.17609 |
| 시작 DFT force에 전역 선형 계수 적용 | 0.09765 | 0.17436 |
| 1-head 직접 endpoint, 3 seeds 평균 ± 표준편차 | 0.05564 ± 0.00055 | 0.10316 ± 0.00323 |
| 4-head 블록, 3 seeds 평균 ± 표준편차 | **0.05370 ± 0.00093** | **0.10146 ± 0.00123** |

4-head의 중간 3개 prefix 위치 오차는 0.03461 ± 0.00057 Å였다. `seed=42`에서 force 입력 없이 같은 4-head 모델을 학습한 ablation은 가동 원자 0.09531 Å, adsorbate 0.16605 Å로 악화됐다. 따라서 이 작은 설정에서 force 정보가 구조 이동 예측의 주요 신호다.

177개 **계 단위**로 윈도우/원자 오차를 먼저 평균한 뒤 직접 endpoint에서 4-head 오차를 뺀 값은 seed 42/43/44에서 각각 `+0.00205`, `+0.00185`, `+0.00016 Å`였다. 계를 재표본한 탐색적 95% bootstrap 구간은 각각 `[+0.00046,+0.00384]`, `[+0.00059,+0.00318]`, `[-0.00053,+0.00086] Å`다. 세 번째 seed에서는 0을 포함하므로 다중 head의 우월성을 강하게 주장할 근거는 없다. Adsorbate 오차의 두 모델 차이도 작다.

## 해석과 다음 검증

이 결과는 **DFT 경로의 네 내부 상태에 공동 감독을 걸면 모델이 학습하고, 직접 endpoint 대비 작은 이득이 나올 수 있다**는 소프트웨어·학습 PoC다. 원 [PDD](https://arxiv.org/html/2607.26004)의 student-state on-policy teacher 호출은 하지 않았다. 연속 DFT 프레임만으로 학생이 경로 밖에서 만든 상태의 DFT 다음 step을 알 수 없기 때문이다.

추가로 실제 OC20 사전학습 MLIP force가 이 DFT-force oracle 결과를 유지하는지, 학생 proposal 후 같은 MLIP로 guard/refinement했을 때 LBFGS보다 **성공률과 최종 구조 품질을 유지하면서 총 wall time이 짧아지는지** 검증해야 한다. 각 run에서 backbone/force/guard/refinement 호출을 전부 비용에 넣는다. 현재 계정의 `facebook/UMA` checkpoint 접근은 401 gated-access로 확인됐다. 아래 실험은 공개 legacy GemNet-OC checkpoint로 대체했다.

## 실제 MLIP를 이용한 8계 CPU 진단

[공식 GemNet-OC-S2EF-OC20-2M checkpoint](https://facebookresearch.github.io/fairchem/models-1/)를 사용했다. 다운로드 URL은 `https://dl.fbaipublicfiles.com/opencatalystproject/models/2022_07/s2ef/gemnet_oc_base_s2ef_2M.pt`, SHA-256은 `a63092b7cf3c42231a5b60e8fb3a33000849e07e8b787698c94e3c1fc76abeca`다. FAIR-Chem v1.10, PyTorch 2.4.1, SciPy 1.14.1을 별도 가상환경에 설치했고, macOS에서 필요한 PyG `torch-scatter`, `torch-sparse`, `torch-cluster`는 [공식 wheel 목록](https://data.pyg.org/whl/torch-2.4.1+cpu.html)에서 설치했다. GemNet-OC는 direct-force 모델이다. Energy 증가 0.1 eV 초과 또는 이동 1 Å 초과 시 제안을 거절하는 guard를 사용했다.

### LBFGS 기준 재실험

처음 FIRE를 기준으로 선택한 것은 잘못이었다. 같은 첫 8개 validation 계에서 ASE LBFGS(`maxstep=0.2 Å`, `memory=100`, `alpha=70`, line search 미사용)를 기준으로 다시 실행했다. 모든 방법은 동일 GemNet checkpoint, 시작 좌표, 고정 원자 제약, `fmax=0.05 eV/Å`, 최대 150 optimizer step을 사용한다. PDD 학생은 작은 새 backbone으로 학습했고, GemNet backbone을 복제한 모델이 아니다. 학생 제안에는 저장 DFT force가 아니라 **GemNet force**를 입력했다. 제안 입력, guard, refinement의 GemNet 평가를 모두 호출 수와 wall time에 포함했다. 모델 로딩은 제외했다.

```bash
MPLCONFIGDIR=data/oc20/cache/mpl XDG_CACHE_HOME=data/oc20/cache/xdg \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  .venv-oc20/bin/python scripts/relaxation_mlip_eval.py \
  --data data/oc20/H_poc_1000 \
  --checkpoint data/oc20/checkpoints/gemnet_oc_base_s2ef_2M.pt \
  --multi-head runs/relaxation/H_1000_L4_3000_force_seed42_multi_head.pt \
  --direct-endpoint runs/relaxation/H_1000_L4_3000_force_seed42_direct_endpoint.pt \
  --output runs/relaxation/gemnet_lbfgs_8x150_one_block.json \
  --systems 8 --max-steps 150 --fmax 0.05 --guard-ev 0.1 \
  --proposal-blocks 1 --proposal-force-source mlip --optimizer lbfgs
```

| 방법 | 수렴 | 총 GemNet 호출 | 총 CPU wall time | LBFGS보다 호출 적은 계 | 최종 에너지 +0.05 eV 초과 |
|---|---:|---:|---:|---:|---:|
| LBFGS | 8/8 | 234 | 109.8 s | — | — |
| 4-head 한 블록 + LBFGS | 8/8 | **207** | 99.8 s | 5/8 | 0/8 |
| 직접 endpoint 한 번 + LBFGS | 8/8 | 213 | **97.7 s** | 6/8 | 0/8 |

한 블록 PDD는 총 호출 수를 27회, 즉 11.5% 줄였다(호출 수 기준 1.13배). 측정 wall time은 1.10배 빨랐지만 CPU 실행 순서와 warmup 영향을 받을 수 있다. 직접 endpoint도 비슷한 수준이므로 이 결과만으로 다중 head의 독자적인 이득은 입증되지 않았다. PDD의 계별 호출 수는 baseline/PDD 순으로 `27/28`, `23/23`, `45/42`, `35/23`, `18/11`, `26/26`, `20/16`, `40/38`이다. PDD 최종 에너지의 LBFGS 대비 차이는 모두 ±0.02 eV 이내였다. 8계 단일 seed 진단이며, 같은 local minimum 및 DFT 구조 정확도를 보장하지 않는다.

두 블록 연속 제안도 동일한 설정에서 `--proposal-blocks 2`로 측정했다. 두 번째 블록의 입력 force는 첫 번째 제안 구조에서 GemNet으로 다시 구한다. 총 비용에 이 평가와 guard 거절도 포함했다.

| 방법 | 수렴 | 총 GemNet 호출 | 총 CPU wall time | LBFGS보다 호출 적은 계 | 최종 에너지 +0.05 eV 초과 |
|---|---:|---:|---:|---:|---:|
| LBFGS | 8/8 | 234 | 108.5 s | — | — |
| 4-head 두 블록 + LBFGS | 8/8 | **198** | **93.4 s** | 6/8 | 0/8 |
| 직접 endpoint 두 번 + LBFGS | 8/8 | 220 | 100.6 s | 5/8 | 0/8 |

두 블록 PDD는 총 호출 수 15.4% 감소(1.18배), 측정 wall time 약 1.16배 단축이었다. PDD 제안 16개 중 14개를 받아들이고 2개를 guard에서 거절했다. 계별 호출 수는 baseline/PDD/direct 순으로 `27/30/39`, `23/21/25`, `45/44/45`, `35/19/25`, `18/11/11`, `26/20/22`, `20/13/14`, `40/40/39`였다. PDD의 최종 에너지는 baseline과 비교해 최대 +0.027 eV였고, 앞선 FIRE 두 블록에서 관찰한 +0.140 eV 다른-basin 사례는 이 LBFGS 설정에서는 재현되지 않았다. 그러나 흡착 원자의 DFT 끝 구조와의 오차는 계마다 증가하기도 했으며, 8계·단일 학습 seed·CPU 실험만으로 품질을 보존한 일반적 가속을 확립할 수 없다. 특히 단일 블록에서는 직접 endpoint와 PDD의 비용 차이가 작고 wall time 순서도 바뀐다. 더 큰 별도 검증집합에서 성공률·에너지·흡착 위치·시간 분포를 확인해야 한다.

### 이전 FIRE 탐색 결과

```bash
UV_CACHE_DIR=.uv-cache uv venv .venv-oc20 --python 3.11
UV_CACHE_DIR=.uv-cache uv pip install --python .venv-oc20/bin/python 'fairchem-core==1.10' ase 'scipy==1.14.1'
UV_CACHE_DIR=.uv-cache uv pip install --python .venv-oc20/bin/python --no-deps \
  --find-links 'https://data.pyg.org/whl/torch-2.4.1+cpu.html' \
  'torch-scatter==2.1.2' 'torch-sparse==0.6.18' 'torch-cluster==1.6.3'
MPLCONFIGDIR=data/oc20/cache/mpl XDG_CACHE_HOME=data/oc20/cache/xdg \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  .venv-oc20/bin/python scripts/relaxation_mlip_eval.py \
  --data data/oc20/H_poc_1000 \
  --checkpoint data/oc20/checkpoints/gemnet_oc_base_s2ef_2M.pt \
  --multi-head runs/relaxation/H_1000_L4_3000_force_seed42_multi_head.pt \
  --direct-endpoint runs/relaxation/H_1000_L4_3000_force_seed42_direct_endpoint.pt \
  --output runs/relaxation/gemnet_cpu_8x80.json \
  --systems 8 --max-steps 80 --fmax 0.05 --guard-ev 0.1 --optimizer fire
```

Hash split의 **첫 8개 validation 계**만 선택했다. 세 방법은 동일 GemNet checkpoint·시작 좌표·제약·FIRE 조건을 공유한다. 학생의 입력 force는 DFT가 아닌 **GemNet force**다. 제안 전 입력 force, proposal guard, optimizer 전 과정의 GemNet 호출과 wall time을 센다. 동일 모델 로딩 시간은 제외한다.

| 방식 | `fmax < 0.05` 수렴 | 8계 총 MLIP 호출 | 8계 총 CPU wall time | FIRE보다 호출 적은 계 |
|---|---:|---:|---:|---:|
| FIRE | 8/8 | 272 | 132.7 s | — |
| 4-head 한 블록 + FIRE | 8/8 | 274 | 126.7 s | 5/8 |
| 직접 endpoint 한 번 + FIRE | 8/8 | 268 | 122.0 s | 5/8 |

입력 구조에서 GemNet과 DFT의 **가동 원자 force 성분 MAE는 평균 0.0417 eV/Å**였다. Force 오차가 작아 보여도 최적화 경로와 수렴 시간까지 같다는 뜻은 아니다. 세 방법은 모두 force 수렴했지만 최종 에너지에 차이가 있으며, 이 8계에서는 제안 방식이 FIRE보다 0.05 eV 넘게 높은 계는 없었다. 한 계에서는 제안 방식이 약 0.2 eV *낮은* local minimum으로 갔다. 작은 표본, CPU 실행 순서와 warmup, direct-force 모델의 energy/force 비보존성 때문에 126.7/132.7초를 유의미한 wall-time 가속으로 해석하지 않는다. **4-head 총 MLIP 호출은 FIRE보다 2회 많고 직접 endpoint보다 6회 많다.** 따라서 현재 설정에서는 PDD relaxation acceleration을 입증하지 못했다.

입력 force 차이를 분리하려고 **저장 DFT force를 proposal 입력에만** 사용하는 비실용적 oracle 진단도 같은 8계에서 수행했다(`--proposal-force-source dft`). FIRE는 272회, 4-head+FIRE는 271회, 직접 endpoint+FIRE는 257회의 MLIP 호출을 썼다. 세 방법 모두 8/8 수렴했다. DFT force를 정확히 줘도 이 모델의 다중 head가 직접 endpoint보다 수렴 비용에서 우월해지지는 않았다. 이 oracle force는 실제 추론에서 무료로 주어지지 않으므로 속도 결과가 아니다.

### 두 블록 연속 적용

PDD 사용 방식에 더 가까운 반복 제안을 `--proposal-blocks 2`로 같은 8계에 시험했다. 두 번째 제안의 force는 첫 제안 구조에서 **GemNet으로 새로 평가**한다. 각 제안의 force/energy 평가와 guard 거절도 비용에 넣는다. FIRE는 항상 같은 시작점에서 실행한다.

| 방식 | `fmax < 0.05` 수렴 | 총 MLIP 호출 | 총 CPU wall time | FIRE보다 호출 적은 계 | FIRE보다 에너지 +0.05 eV 초과 |
|---|---:|---:|---:|---:|---:|
| FIRE | 8/8 | 272 | 129.3 s | — | — |
| 4-head 두 블록 + FIRE | 8/8 | **231** | **106.0 s** | 5/8 | **1/8** |
| 직접 endpoint 두 번 + FIRE | 8/8 | 257 | 116.4 s | 6/8 | **1/8** |

4-head는 두 제안 중 총 14개 블록을 받아들였고 2개를 거절했다. 계별 호출 수는 baseline/PDD/direct 순으로 `19/29/49`, `32/42/34`, `44/58/43`, `35/20/33`, `29/14/14`, `40/26/39`, `33/20/20`, `40/22/25`였다. 초기 세 계의 손해가 크고 나머지 다섯 계는 이득이 있다. 마지막 계에서는 PDD의 최종 에너지가 baseline보다 **0.140 eV 높고**, DFT 최종 흡착 원자와의 거리가 **0.835 Å**로 baseline **0.064 Å**보다 훨씬 크다. 한 블록만 썼을 때 같은 계의 PDD 최종 에너지 차이는 약 -0.0005 eV였다. 즉 **두 번째 제안이 다른 basin으로 보낸 사례**다. 단순 에너지 증가 guard는 각 이동을 허용했지만 이 문제를 막지 못했다.

이 소규모 CPU 진단에서 두 블록은 **계산비를 줄일 가능성**을 보였지만, 성공 정의를 `fmax` 하나로 놓으면 잘못된 결론에 이른다. 같은 local minimum 또는 사전 정의한 에너지·흡착 위치 허용오차를 유지하는지까지 검증해야 한다. Direct-force GemNet의 energy/force 일관성도 별도 제약이다. 모델은 작은 새 backbone으로 학습했으며 원 PDD의 on-policy 학생 상태 감독이나 GemNet pretrained backbone 복제는 아직 적용하지 않았다. 따라서 실패 사례를 PDD 접근 전체의 한계로 일반화하지 않는다.
