# PDD를 MLIP 구조 relaxation에 적용할 수 있는가?

2026-09-28 조사 메모. 결론: **시험할 가치가 높고, 첫 유용한 결과는 MD보다 빨리 얻을 가능성이 있다.** 목적을 `같은 MLIP의 local relaxation을 더 적은 wall time에 완료`로 좁혀야 한다. 공개 DFT 궤적 자체를 그대로 PDD teacher 경로라고 부르거나, MLIP force가 작은 구조를 곧바로 DFT 최소점이라고 주장하면 안 된다. 아래의 장점과 한계는 [PDD 원 논문](https://arxiv.org/html/2607.26004), [OC20 공식 데이터 문서](https://facebookresearch.github.io/fairchem/oc20/), [ASE optimizer 문서](https://ase.gitlab.io/ase/ase/optimize.html), [FAIR-Chem 모델 문서](https://facebookresearch.github.io/fairchem/models-1/)를 바탕으로 한 연구 판단이다.

## 왜 MD보다 맞을 수 있나

| 관점 | MD | Local relaxation |
|---|---|---|
| 필요한 결과 | 긴 경로 또는 시간 통계 | 힘이 작은 유효 구조와 충분히 낮은 에너지 |
| 누적 오차 | 초기 오차가 궤적 전체에 증폭될 수 있음 | 마지막에 기존 optimizer로 보정 가능 |
| 공개 경로 | AD-3는 train/test 각각 한 긴 궤적 | OC20은 수십만 개 서로 다른 계의 relaxation 경로 |
| 성능 비교 | 물리적 안정성과 통계 샘플링까지 요구 | 동일 포텐셜·수렴 기준에서 호출 수/시간/성공률 비교 가능 |

이는 *추론*이다. Relaxation은 최종 상태가 중요해서 여러 내부 상태를 한 번에 예측하는 PDD 구조의 이점이 오히려 작을 수도 있다. 단일 endpoint predictor나 큰 optimizer step이 더 간단하고 강한 경쟁자다. PDD 고유 이득은 `같은 계산비의 직접 endpoint 예측보다 낮은 실패율 또는 적은 최종 refinement 호출`로 입증해야 한다.

## 데이터는 실제로 얼마나 있고 무엇인가

[OC20 논문](https://arxiv.org/abs/2010.09990)은 전체 생성 규모를 약 128만 DFT relaxation과 2억 6천만 단일점 평가로 보고한다. **현재 공개 IS2RE/IS2RS 학습용 relaxation 궤적 묶음**은 약 46.6만 경로이며, [공식 다운로드 표](https://facebookresearch.github.io/fairchem/oc20/)의 전체 묶음은 압축 109 GB, 해제 후 841 GB다. 공식 validation은 ID/OOD-adsorbate/OOD-catalyst/OOD-both 각각 약 2.5만 경로이며 압축 4.4–6.0 GB다. 전체 규모와 지금 당장 학습에 필요한 파일 크기를 혼동하지 않는다. S2EF의 200K/2M frame subset은 **경로 전체 묶음과 다르다**.

빠른 파일·파서 PoC에는 같은 [공식 문서의 per-adsorbate trajectory](https://facebookresearch.github.io/fairchem/oc20/)를 쓴다. 예를 들어 `*H`는 약 850 MB, `*O`는 약 1006 MB, `*OH`는 약 1.6 GB인 개별 tar다. 처음에는 `*OH` 1개와 공식 validation의 작은 고정 subset을 권한다. 단일 adsorbate만으로 일반화 주장을 하지 않는다. 다운로드는 공식 표의 해당 링크·MD5로 확인하고, tar 구성과 실제 step 수·힘·에너지·PBC·fixed atom mask·system ID를 먼저 샘플 조사한다. 전체 109 GB를 미리 받을 필요는 없다.

OC20 DFT의 ASE 궤적은 **DFT 최적화기의 결과**다. 지금 목표인 `특정 MLIP + 특정 optimizer`의 경로와는 일반적으로 다르며, optimizer 내부 상태도 프레임만으로 항상 복구되는 것은 아니다. 따라서 세 데이터 사용법을 명시적으로 분리한다.

1. **주 실험:** OC20의 구조를 *초기상태*로 쓰고 고정한 MLIP와 optimizer로 teacher relaxation을 새로 생성한다. Student가 만든 좌표에서 같은 MLIP를 호출해 on-policy PDD target을 계산한다.
2. **선택적 사전학습:** 공개 DFT 궤적의 `x_t → x_{t+k}`와 마지막 구조를 offline imitation에 사용한다. DFT와 MLIP의 최소점이 다를 수 있으므로 주 실험의 정답으로 섞지 않는다.
3. **DFT 수준 검증:** 최종 구조에 DFT 단일점 force/energy 또는 refinement를 수행할 수 있을 때만 DFT relaxation 가속 또는 DFT 정확도를 주장한다. 이것은 별도 예산이 드는 후속 실험이다.

OC20 표준 split은 동일 분자·슬랩 계의 frame이 train/validation에 섞이지 않게 유지하고, ID와 세 OOD split을 따로 보고한다. 선택한 MLIP가 OC20 train을 이미 본 pretrained 모델이면 teacher/student의 OC20 validation은 구조 일반화 평가가 되지만, Student를 학습시키는 DFT 궤적과 model-pretraining의 데이터 출처도 기록한다. [OC20-Dense](https://facebookresearch.github.io/fairchem/oc20dense/)는 여러 초기 배치를 통한 **global adsorption minimum** 평가용이다. 우리의 단일 시작점 local minimization 지표와 구분하고, 나중에 AdsorbML 형태의 평가를 할 때 사용한다.

## 모델과 teacher 선택

OMol eSEN 체크포인트를 OC20 슬랩에 그대로 적용하지 않는다. 첫 후보는 [FAIR-Chem의 `UMA` + `oc20` task](https://github.com/facebookresearch/fairchem)다. 공식 [UMA 문서](https://facebookresearch.github.io/fairchem/uma/)는 energy-conserving 모델과 `oc20` task를 제공하고, [relaxation 예제](https://github.com/facebookresearch/fairchem)에는 `FAIRChemCalculator(..., task_name="oc20")`와 ASE LBFGS가 있다. 다만 이 repo의 현재 eSEN adapter를 UMA에 그대로 연결할 수 있다는 증거는 없다. UMA는 routing/출력 head 구조가 다르므로, frozen teacher 로딩·batch force/energy·student head 복제·PBC·fixed mask를 짧은 기술 선행시험으로 검증한다.

구현 난도가 높으면 별도 legacy 환경의 [OC20 학습 GemNet-OC-2M 또는 eSCN](https://facebookresearch.github.io/fairchem/models-1/)으로 먼저 알고리즘을 검증한다. 공식 문서는 이 체크포인트들이 FAIRChem v1용이며 v2와 호환되지 않는다고 명시한다. current MD repo의 `fairchem-core 2.23` 환경에 무작정 v1 모델을 로드하지 않는다. 교차 비교 시 같은 teacher/checkpoint와 같은 포텐셜 안에서만 속도·정확도를 짝지어 비교한다. 최신 UMA를 쓸 수 있다면 그것이 production 쪽 첫 선택이다.

Teacher optimizer는 먼저 **FIRE 또는 LBFGS**를 고정하고 ASE 구현·수렴 threshold·최대 step·step cap·line search 설정을 저장한다. [ASE 문서](https://ase.gitlab.io/ase/ase/optimize.html)의 BFGS/LBFGS는 Hessian 또는 history, FIRE는 속도·시간 간격 등의 상태를 갖는다. 좌표만 보고 같은 다음 optimizer step을 정의할 수 없으므로 PDD를 정확히 이식하려면 확장 상태 `(q, optimizer_state)`를 모델 입력/teacher map에 포함하거나, 첫 PoC에는 메모리 없는 fixed-step force descent를 teacher로 정의해야 한다. 후자는 공정한 최고 성능 baseline이 아니므로 반드시 FIRE/LBFGS와도 비교한다.

## 권장 방법: PDD block proposal + 기존 optimizer refinement

1. 학생 입력: 원자 종류·좌표·unit cell/PBC·adsorbate/slab tag·fixed-atom mask와 필요한 optimizer state. OC20에서 고정된 원자는 절대 움직이지 않고, 주기 경계를 올바르게 다룬다. 첫 PoC는 cell 고정 relaxation으로 제한한다.
2. 한 backbone 평가에서 `L=2 또는 4`개 displacement head를 예측하고 내부 좌표 `q_1,...,q_L`을 구성한다. 각 head는 teacher의 해당 한 step displacement 또는 상태 변화량을 학생이 만든 내부 상태에서 감독한다. 이 부분이 [PDD](https://arxiv.org/html/2607.26004)의 핵심이고, 저장 DFT 경로를 단순히 L칸 건너뛰어 회귀하는 것과 다르다.
3. 최종 `q_L`에서 **같은 MLIP**의 energy/force를 다시 평가한다. 큰 충돌, 지나친 이동, 에너지 증가, 힘 발산이면 jump를 거절하거나 줄인다. Guard 호출도 비용에 포함한다.
4. 충분히 가깝다면 FIRE/LBFGS로 기존 수렴 조건까지 refinement한다. 속도는 `proposal + guard + refinement` 전체 시간으로 재야 한다. 실패한 계도 fallback을 돌려 성공률과 비용에 넣는다.

첫 head는 teacher force 쪽 출력을 복제해 근접한 업데이트로 초기화할 수 있지만, force 단위(eV/Å)와 displacement 단위(Å)가 다르므로 정규화/step-size 변환이 필수다. 현재 MD용 qdot/acceleration head를 그대로 복제해서 쓰지는 않는다. PDD 최종 출력 자체가 구조 전체를 예측한다면, **직접 endpoint head + 동일 refinement**가 꼭 필요한 ablation이다. 처음에는 온폴리시 teacher를 각 배치에 호출하고, DFT 궤적 supervision은 부가 loss로만 시험한다.

## 공정한 평가와 필수 baseline

모든 방법은 **동일 MLIP, 같은 입력 구조, 같은 PBC/constraint, 같은 수렴 조건**을 공유한다. 기본 조건의 예는 가동 원자의 최대 force `fmax < 0.05 eV/Å`와 고정 최대 평가 수이며, 실제 threshold는 pilot 전에 정한다. force-only 모델은 energy가 보존적이지 않을 수 있으니 에너지 기준을 쓰기 전에 energy–force 일관성을 점검한다.

| 방법 | 이유 |
|---|---|
| MLIP + ASE FIRE | 견고한 고전적 기준 |
| MLIP + ASE LBFGS/BFGS | 빠른 국소 수렴 기준 |
| trust radius/step cap을 조정한 FIRE/LBFGS | 약한 기본 설정을 이기는 착시 방지 |
| 직접 `x_0 → x_*` 또는 L-step endpoint 모델 + 동일 refinement | PDD multi-head가 필요한지 검증 |
| PDD L2/L4 + guard + 동일 refinement | 제안 방법 |
| 가능하면 공개 DFT 궤적을 이용한 offline multi-step predictor | 데이터 활용 방식 비교 |

기록: 초기/최종/최대 `fmax`, MLIP energy 차이, 수렴·실패·fallback 비율, 원자 충돌/탈착/표면 재구성 이상률, relaxed 좌표와 adsorbate 위치 오차, reference와 다른 basin 비율, 총 backbone·force·energy 평가 수, 중간 line-search·guard 호출 수, GPU wall time p50/p90, GPU peak memory, batch throughput, 시스템 크기·adsorbate별 성능. 실패를 제외한 평균 속도는 제시하지 않는다. 동일 MLIP로 끝까지 수렴한 구조를 reference로 삼되, 더 낮은 local minimum을 찾은 경우를 무조건 오답 처리하지 않는다. DFT 구조와 비교할 때는 MLIP 포텐셜 차이를 별도 오차로 기록한다. OC20의 IS2RS/IS2RE 지표도 계산하되, local 최적화 비용을 가리지 않는다.

`L=4`가 네 번의 optimizer step을 하나의 backbone 호출로 바꾸더라도, 추가 guard/line search/refinement가 있으면 4배 wall-time 이득은 보장되지 않는다. 비용이 비슷하다는 단순 예로 원래 `N`번의 force 호출에 비해 proposal+guard가 block당 두 번이고 refinement가 `R`번이면 호출 수 비는 대략 `N/(2N/L+R)`다. 실제로는 UMA의 energy gradient와 student forward 시간이 달라 반드시 측정한다.

## 4×3090에서의 첫 2주 계획

**1–2일:** 공식 per-adsorbate `*OH` tar와 작은 ID/OOD validation subset을 받기 전에 서버 저장 공간을 확인한다. 파일 hash와 split·trajectory ID를 검증하고 ASE로 100개 경로의 길이, force, 움직이는 atom 수를 조사한다. UMA `oc20` predictor의 force/energy, PBC, constraint, batch 및 GPU 메모리를 점검한다. 이 모델에서 에너지/force를 얻는 방법과 head 복제 가능성을 코드로 증명한다.

**3–5일:** 1 GPU에서 100–1,000개 초기 구조를 동일 MLIP로 FIRE와 LBFGS relaxation해 경로와 wall time을 저장한다. 나머지 GPU는 별도 구조 배치의 teacher 생성에 사용한다. 전체 OC20를 내려받거나 새로운 DFT 계산을 시작하지 않는다. 검증/최종 test의 system ID를 고정한다.

**6–10일:** 작은 `L=2/4` 모델, 직접 endpoint baseline, guard+refinement 파이프라인을 구현한다. 4 GPU는 독립 seed·L·batch 실험에 먼저 쓴다. teacher path를 캐시해 offline ablation을 하고, 본 PDD는 학생 내부 상태에서 같은 MLIP teacher를 조회한다. 과대 batch와 loss weight를 늘리기 전에 한 head·한 block의 target/parity를 검증한다.

**11–14일:** ID와 OOD validation에서 같은 `fmax`의 성공률 대 wall-time 및 평가 횟수 곡선을 비교한다. PDD가 조정된 FIRE/LBFGS 및 direct endpoint보다 낫지 않거나 head 복제 비용이 너무 높으면 그 결과를 기록하고 MD와 별개 연구 방향으로 유지한다. 기초 비교에서 우월한 영역이 보이면 OC20 전체 train과 OOD 체계, OC20-Dense/DFT 후속 평가로 확장한다. DDP는 단일 GPU 병목·모델 구조를 확인한 뒤 구현한다.

## 연구 포지셔닝

문헌상 MLIP를 이용한 relaxation 가속 자체는 새롭지 않다. [AdsorbML](https://www.nature.com/articles/s41524-023-01121-5)은 MLIP로 후보를 relax한 뒤 DFT 단일점/재최적화를 결합했고, [AdsorbDiff](https://proceedings.mlr.press/v235/kolluru24a.html)와 [AdsorbFlow](https://arxiv.org/abs/2602.19289)는 낮은 에너지 adsorbate 배치를 생성한다. 따라서 새로운 주장은 **OC20 규모의 relaxation 경로에서 pretrained potential의 여러 optimizer 변화를 한 번의 shared backbone 평가로 내고, 같은 포텐셜의 수렴 품질을 보존하면서 총 relaxation 시간을 단축한다**는 것으로 한정하는 편이 타당하다. 물리적 수렴과 벽시계 시간에서 경쟁 기준을 이겨야 한다. 이 프로젝트의 기존 MD 코드는 데이터 로더·위상공간 head·교사용 적분·PBC 가정이 달라 relaxation에 그대로 적용할 수 없고, 공통 backbone/다중 head/PDD 감독 아이디어만 재사용한다.
