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

## 100계 두 블록 결과

100계 × 3방법의 **300개 실행이 모두 완료**됐다. CPU 실험의 측정 wall time 합계이며 모델 로딩은 제외한다.

| 방법 | `fmax < 0.05` 수렴 | GemNet 호출 합계 | wall time 합계 | LBFGS보다 호출 적은 계 | 기준보다 에너지 +0.05 eV 초과 | DFT 흡착 원자 오차 +0.20 Å 초과 |
|---|---:|---:|---:|---:|---:|---:|
| LBFGS | **100/100** | 4,377 | 2,450.6 s | — | — | — |
| PDD 4-head 두 블록 + LBFGS | **98/100** | 4,302 | 2,375.9 s | 65/100 | 10/98 | 7/98 |
| 직접 endpoint 두 번 + LBFGS | **98/100** | **4,123** | **2,371.6 s** | 64/100 | 8/98 | 7/98 |

PDD는 총 호출 수를 **1.7%** 줄였지만, 계 단위 paired bootstrap 95% 구간은 **-8.2%에서 +10.1%**다. 측정 wall time 절감률은 **3.0%**이고 같은 방식의 구간은 **-6.3%에서 +11.3%**다. 직접 endpoint는 호출 수 **5.8%**, wall time **3.2%** 절감했다. 두 방법 모두 기준 LBFGS의 100/100 수렴률을 유지하지 못했다. PDD와 직접 endpoint의 wall time 합계 차이는 4.3초로, 약 2,400초 규모에서 다중 head의 계산상 우위를 뒷받침하지 않는다.

PDD는 65계에서 호출 수가 줄었고 8계는 같으며 27계에서는 늘었다. 증가가 큰 계의 초과 호출은 `random1138188` +102회, `random2275030` +94회, `random1408921` +76회다. 이 세 계가 소수 실패의 비용 위험을 보여준다. PDD의 최대 최종 에너지 악화는 **+3.819 eV**였고, DFT 최종 흡착 위치 오차의 최대 증가는 **+2.896 Å**였다. 최종 에너지가 오히려 낮으면서 흡착 위치 오차가 2 Å 이상 커진 계도 있어 energy guard만으로 구조 품질을 판정할 수 없다. PDD 블록 제안 200개 중 185개를 받아들였고 13계에서 제안을 거절했다. 거절 횟수는 계별 boolean으로 기록되어 있으므로 여러 제안 거절 횟수와 동일하지 않다.

**판정:** 이 100계 평가에서는 품질을 유지하는 PDD relaxation 가속이 입증되지 않았다. 8계 pilot의 큰 이득이 대표적이지 않았고, 다중 head가 직접 endpoint를 안정적으로 이기지 못했다. 특히 GemNet은 direct-force 모델이라 energy와 force가 엄밀하게 보존 관계에 있다는 보장은 없다. `fmax` 수렴과 에너지·흡착 구조를 함께 평가해야 한다. 이 결론은 OC20 `*H`의 작은 동일 아카이브 계에 한정되며, 다른 adsorbate나 공식 OOD로 일반화하지 않는다.

실행 원시값은 로컬 `runs/relaxation/gemnet_lbfgs_holdout100_two_blocks.json`에 남겼다. 이를 사용해 위 표와 paired bootstrap 구간을 `scripts/summarize_relaxation_eval.py`로 재생성할 수 있다. 큰 checkpoint와 원시 JSON은 Git에 올리지 않는다.

## 탐색적 한 블록 ablation

두 블록 결과에서 비수렴과 큰 구조 오차가 드러난 **뒤에** 같은 100계의 한 블록만 시험하기로 했다. 따라서 이는 위 100계의 독립 확증 결과가 아니라 사후 원인 탐색이다. 학생 checkpoint, MLIP, guard, LBFGS 설정을 바꾸지 않고 `--proposal-blocks 1`을 적용한다. 동일 시작점의 LBFGS baseline은 위 원시 JSON의 100개 결과를 그대로 재사용한다.

```bash
MPLCONFIGDIR=data/oc20/cache/mpl XDG_CACHE_HOME=data/oc20/cache/xdg \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  .venv-oc20/bin/python scripts/relaxation_mlip_eval.py \
  --data data/oc20/H_poc_1000 \
  --checkpoint data/oc20/checkpoints/gemnet_oc_base_s2ef_2M.pt \
  --multi-head runs/relaxation/H_1000_L4_3000_force_seed42_multi_head.pt \
  --direct-endpoint runs/relaxation/H_1000_L4_3000_force_seed42_direct_endpoint.pt \
  --output runs/relaxation/gemnet_lbfgs_holdout100_one_block.json \
  --skip-systems 8 --systems 100 --max-steps 150 --fmax 0.05 \
  --guard-ev 0.1 --proposal-blocks 1 --proposal-force-source mlip \
  --optimizer lbfgs --methods pdd direct
.venv/bin/python scripts/summarize_relaxation_eval.py \
  runs/relaxation/gemnet_lbfgs_holdout100_one_block.json \
  --baseline-from runs/relaxation/gemnet_lbfgs_holdout100_two_blocks.json
```

100계 × 2방법의 **200개 실행이 모두 완료**됐고 두 블록 평가와 계 ID 100개가 정확히 일치한다. 아래 기준 LBFGS 호출 수와 품질은 두 블록 실험 원시 JSON의 같은 계 결과를 재사용했다. 따라서 호출 수 비교는 가능하지만, **서로 다른 실행 시점의 CPU wall time을 나눈 가속 배수는 보고하지 않는다.** 한 블록 실행의 원시 wall time 합계는 PDD 2,236.8초, 직접 endpoint 2,235.6초였다.

| 방법 | `fmax < 0.05` 수렴 | GemNet 호출 합계 | LBFGS보다 호출 적은 계 | 기준보다 에너지 +0.05 eV 초과 | DFT 흡착 원자 오차 +0.20 Å 초과 |
|---|---:|---:|---:|---:|---:|
| 재사용한 LBFGS 기준 | **100/100** | 4,377 | — | — | — |
| PDD 4-head 한 블록 + LBFGS | 99/100 | 4,251 | 63/100 | 7/99 | 7/99 |
| 직접 endpoint 한 번 + LBFGS | 99/100 | **4,240** | 57/100 | 8/99 | 6/99 |

한 블록 PDD의 총 호출 절감률은 **2.9%**, 계 단위 paired bootstrap 95% 구간은 **-6.7%에서 +10.6%**다. 직접 endpoint는 3.1%, 구간 **-6.3%에서 +11.1%**였다. PDD와 직접 endpoint의 호출 합계 차이는 11회(100계 전체)로 작고, PDD가 적은 계 43개·동률 26개·많은 계 31개다. 다중 head의 독자적인 이득은 보이지 않는다.

두 블록 대비 PDD 수렴이 98→99계로 늘고, 에너지 +0.05 eV 초과 계는 10→7계로 줄었다. 그러나 한 블록에서도 최대 에너지 악화는 **+3.789 eV**, DFT 흡착 위치 오차 증가는 **+2.890 Å**였다. `random2275030`은 두 블록에서 PDD 118회·흡착 오차 1.668 Å였으나 한 블록에서는 25회·0.179 Å로 기준(24회·0.153 Å)에 가까워졌다. 반면 `random1143732`는 한 블록에서도 흡착 오차 2.924 Å로 기준 0.034 Å와 크게 달랐고, `random249251`의 약 +3.8 eV 다른 minimum도 남았다. 따라서 두 번째 제안을 생략하면 일부 실패는 줄지만 첫 제안에서의 다른-basin 이동은 해결되지 않는다.

**종합:** 제안 횟수를 한 번 또는 두 번으로 바꿔도 이 오프라인 학생 모델은 기준 LBFGS의 100/100 수렴과 구조 품질을 재현하지 못한다. 호출 수 절감의 bootstrap 구간도 두 설정 모두 0을 포함한다. 이 결과는 원 PDD의 학생 방문 상태 감독이나 pretrained GemNet backbone 복제를 시험한 것이 아니므로, 그 설계 전체의 부정적 결론은 아니다. 다음 구현 단계는 첫 제안 자체의 신뢰도·force/energy 일관성 검증과 on-policy 보강이며, 큰 서버 실험 전에 공식 OC20 validation의 더 다양한 흡착종·계로 분할을 확장해야 한다.
