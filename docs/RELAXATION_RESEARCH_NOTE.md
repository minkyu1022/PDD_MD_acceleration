# PDD를 MLIP 구조 relaxation에 적용할 수 있는가?

2026-09-28 조사 메모. 결론: **시험할 가치가 높고, 첫 유용한 결과는 MD보다 빨리 얻을 가능성이 있다.** 핵심 가정은 *OC20의 DFT relaxation 경로가 선택한 MLIP에서의 구조 이동을 학습하기에 충분히 가깝다*는 것이다. 따라서 공개 DFT 궤적을 **주 학습 데이터**로 쓰고, 추론 중 MLIP 평가와 최종 보정을 결합해 relaxation 시간을 줄인다. MLIP로 학습용 전체 경로를 재생성할 필요는 없다. 이 가정의 성립 범위는 소규모 경로 비교와 최종 수렴 평가로 확인한다. 아래의 장점과 한계는 [PDD 원 논문](https://arxiv.org/html/2607.26004), [OC20 공식 데이터 문서](https://facebookresearch.github.io/fairchem/oc20/), [ASE optimizer 문서](https://ase.gitlab.io/ase/ase/optimize.html), [FAIR-Chem 모델 문서](https://facebookresearch.github.io/fairchem/models-1/)를 바탕으로 한 연구 판단이다.

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

OC20의 저장 궤적은 **DFT 최적화기의 결과**다. 이를 MLIP relaxation의 근사 경로로 취급하는 것이 이 프로젝트의 출발점이다. 학습용 MLIP 궤적을 모두 다시 만들지 않고 다음 순서로 시험한다.

1. **주 학습:** 공개 DFT 궤적의 연속 프레임으로 `x_t → (x_{t+1},...,x_{t+L})`의 다중 head 변화를 감독한다. 궤적 단위 split을 지킨다. DFT 계산이나 MLIP 경로 재생성은 필요하지 않다.
2. **PDD on-policy 보강:** 원 PDD처럼 *student가 만든 중간 구조*에서 target을 얻고 싶을 때만, 그 구조에 선택한 MLIP를 호출해 짧은 local update를 만든다. 저장 DFT 프레임만으로는 경로 밖 임의 구조의 DFT 다음 step을 알 수 없다. 이 보강은 전체 MLIP relaxation 경로를 생성하는 일과 다르며, 비용과 효과를 별도 ablation으로 기록한다.
3. **평가:** 같은 시작 구조에서 MLIP FIRE/LBFGS baseline과 student proposal+refinement를 실제 실행한다. DFT 끝 구조와의 거리 및 MLIP 최종 force/energy를 함께 기록해 DFT↔MLIP 근사가 어디서 깨지는지 본다. DFT 수준 수렴 주장은 DFT 검증이 있을 때만 한다.

OC20 표준 split은 동일 분자·슬랩 계의 frame이 train/validation에 섞이지 않게 유지하고, ID와 세 OOD split을 따로 보고한다. 선택한 MLIP가 OC20 train을 이미 본 pretrained 모델이면 teacher/student의 OC20 validation은 구조 일반화 평가가 되지만, Student를 학습시키는 DFT 궤적과 model-pretraining의 데이터 출처도 기록한다. [OC20-Dense](https://facebookresearch.github.io/fairchem/oc20dense/)는 여러 초기 배치를 통한 **global adsorption minimum** 평가용이다. 우리의 단일 시작점 local minimization 지표와 구분하고, 나중에 AdsorbML 형태의 평가를 할 때 사용한다.

## 모델과 teacher 선택

OMol eSEN 체크포인트를 OC20 슬랩에 그대로 적용하지 않는다. 첫 후보는 [FAIR-Chem의 `UMA` + `oc20` task](https://github.com/facebookresearch/fairchem)다. 공식 [UMA 문서](https://facebookresearch.github.io/fairchem/uma/)는 energy-conserving 모델과 `oc20` task를 제공하고, [relaxation 예제](https://github.com/facebookresearch/fairchem)에는 `FAIRChemCalculator(..., task_name="oc20")`와 ASE LBFGS가 있다. 2026-09-28 현재 이 Mac의 Hugging Face 계정으로 `facebook/UMA` checkpoint metadata 요청은 401 gated-access 오류를 반환했다. 따라서 UMA를 쓰는 후속 실험에는 별도 접근 승인이 필요하다. 이 repo의 현재 eSEN adapter를 UMA에 그대로 연결할 수 있다는 증거도 없다. UMA는 routing/출력 head 구조가 다르므로, frozen model 로딩·batch force/energy·student head 복제·PBC·fixed mask를 짧은 기술 선행시험으로 검증한다.

구현 난도가 높으면 별도 legacy 환경의 [OC20 학습 GemNet-OC-2M 또는 eSCN](https://facebookresearch.github.io/fairchem/models-1/)으로 먼저 알고리즘을 검증한다. 공식 문서는 이 체크포인트들이 FAIRChem v1용이며 v2와 호환되지 않는다고 명시한다. current MD repo의 `fairchem-core 2.23` 환경에 무작정 v1 모델을 로드하지 않는다. 교차 비교 시 같은 teacher/checkpoint와 같은 포텐셜 안에서만 속도·정확도를 짝지어 비교한다. 최신 UMA를 쓸 수 있다면 그것이 production 쪽 첫 선택이다.

공개 DFT 경로의 연속 프레임을 감독할 때는 원래 DFT optimizer의 내부 상태를 복원할 필요가 없다. 하지만 저장 프레임의 `x_t → x_{t+1}`은 일반적인 좌표 전이 함수가 아니다. [ASE 문서](https://ase.gitlab.io/ase/ase/optimize.html)의 BFGS/LBFGS는 Hessian 또는 history, FIRE는 속도·시간 간격 등의 상태를 갖는다. 따라서 이 오프라인 모델은 **저장된 경로의 변화를 예측**하며, 경로 밖 student 상태에 대한 정확한 DFT optimizer step을 모방한다고 주장하지 않는다. MLIP on-policy 보강을 할 때는 메모리 없는 명시적 local update를 정의하거나 `(q, optimizer_state)`를 함께 다룬다.

## 권장 방법: PDD block proposal + 기존 optimizer refinement

1. 학생 입력: 원자 종류·좌표·unit cell/PBC·adsorbate/slab tag·fixed-atom mask와 필요한 optimizer state. OC20에서 고정된 원자는 절대 움직이지 않고, 주기 경계를 올바르게 다룬다. 첫 PoC는 cell 고정 relaxation으로 제한한다.
2. 한 backbone 평가에서 `L=2 또는 4`개 displacement head를 예측하고 내부 좌표 `q_1,...,q_L`을 구성한다. 먼저 OC20 저장 경로의 **각 중간 프레임**을 감독한다. 이는 PDD의 다중 head/block 구조를 활용한 오프라인 버전이다. 원 [PDD](https://arxiv.org/html/2607.26004)의 on-policy 감독까지 검증하려면, student 내부 구조에서 MLIP local update를 추가로 평가하는 실험을 분리한다.
3. 최종 `q_L`에서 **같은 MLIP**의 energy/force를 다시 평가한다. 큰 충돌, 지나친 이동, 에너지 증가, 힘 발산이면 jump를 거절하거나 줄인다. Guard 호출도 비용에 포함한다.
4. 충분히 가깝다면 FIRE/LBFGS로 기존 수렴 조건까지 refinement한다. 속도는 `proposal + guard + refinement` 전체 시간으로 재야 한다. 실패한 계도 fallback을 돌려 성공률과 비용에 넣는다.

첫 head는 MLIP force 쪽 출력을 복제해 근접한 업데이트로 초기화할 수 있지만, force 단위(eV/Å)와 displacement 단위(Å)가 다르므로 정규화/step-size 변환이 필수다. 현재 MD용 qdot/acceleration head를 그대로 복제해서 쓰지는 않는다. PDD 최종 출력 자체가 구조 전체를 예측한다면, **직접 endpoint head + 동일 refinement**가 꼭 필요한 ablation이다. 첫 학습은 공개 DFT 궤적만 사용하고, on-policy MLIP 보강은 성능을 보고 추가한다.

## 공정한 평가와 필수 baseline

모든 방법은 **동일 MLIP, 같은 입력 구조, 같은 PBC/constraint, 같은 수렴 조건**을 공유한다. 기본 조건의 예는 가동 원자의 최대 force `fmax < 0.05 eV/Å`와 고정 최대 평가 수이며, 실제 threshold는 pilot 전에 정한다. force-only 모델은 energy가 보존적이지 않을 수 있으니 에너지 기준을 쓰기 전에 energy–force 일관성을 점검한다.

| 방법 | 이유 |
|---|---|
| MLIP + ASE FIRE | 견고한 고전적 기준 |
| MLIP + ASE LBFGS/BFGS | 빠른 국소 수렴 기준 |
| trust radius/step cap을 조정한 FIRE/LBFGS | 약한 기본 설정을 이기는 착시 방지 |
| 직접 `x_0 → x_*` 또는 L-step endpoint 모델 + 동일 refinement | PDD multi-head가 필요한지 검증 |
| PDD L2/L4 + guard + 동일 refinement | 제안 방법 |
| 공개 DFT 궤적을 이용한 direct endpoint predictor | multi-head 구조의 추가 이득 비교 |
| 선택적 MLIP on-policy 보강 | 오프라인 학습과 경로 이탈 보강의 차이 측정 |

기록: 초기/최종/최대 `fmax`, MLIP energy 차이, 수렴·실패·fallback 비율, 원자 충돌/탈착/표면 재구성 이상률, relaxed 좌표와 adsorbate 위치 오차, reference와 다른 basin 비율, 총 backbone·force·energy 평가 수, 중간 line-search·guard 호출 수, GPU wall time p50/p90, GPU peak memory, batch throughput, 시스템 크기·adsorbate별 성능. 실패를 제외한 평균 속도는 제시하지 않는다. 동일 MLIP로 끝까지 수렴한 구조를 reference로 삼되, 더 낮은 local minimum을 찾은 경우를 무조건 오답 처리하지 않는다. DFT 구조와 비교할 때는 MLIP 포텐셜 차이를 별도 오차로 기록한다. OC20의 IS2RS/IS2RE 지표도 계산하되, local 최적화 비용을 가리지 않는다.

`L=4`가 네 번의 optimizer step을 하나의 backbone 호출로 바꾸더라도, 추가 guard/line search/refinement가 있으면 4배 wall-time 이득은 보장되지 않는다. 비용이 비슷하다는 단순 예로 원래 `N`번의 force 호출에 비해 proposal+guard가 block당 두 번이고 refinement가 `R`번이면 호출 수 비는 대략 `N/(2N/L+R)`다. 실제로는 UMA의 energy gradient와 student forward 시간이 달라 반드시 측정한다.

## 4×3090에서의 첫 2주 계획

**1–2일:** 공식 per-adsorbate `*OH` tar와 작은 ID/OOD validation subset을 받기 전에 서버 저장 공간을 확인한다. 파일 hash와 split·trajectory ID를 검증하고 ASE로 100개 경로의 길이, force, 움직이는 atom 수를 조사한다. UMA `oc20` predictor의 force/energy, PBC, constraint, batch 및 GPU 메모리를 점검한다. 이 모델에서 에너지/force를 얻는 방법과 head 복제 가능성을 코드로 증명한다.

**3–5일:** OC20 DFT 경로에서 `(x_t,...,x_{t+L})` 윈도우를 만들고, 작은 고정 subset에서 다중 head가 각 프레임을 재현하는지 확인한다. 별도로 100–1,000개 검증 초기 구조에서 동일 MLIP의 FIRE/LBFGS를 돌려 기준 시간·수렴률과 DFT 끝 구조와의 차이를 측정한다. 이 평가 subset 외에 학습용 MLIP 경로를 대량으로 재생성하지 않는다. 검증/최종 test의 system ID를 고정한다.

**6–10일:** 작은 `L=2/4` 모델, 직접 endpoint baseline, guard+refinement 파이프라인을 구현한다. 4 GPU는 독립 seed·L·batch 실험에 먼저 쓴다. 주 학습은 DFT 경로 supervision이고, MLIP on-policy target은 별도 ablation으로 추가한다. 과대 batch와 loss weight를 늘리기 전에 한 head·한 block의 target/parity를 검증한다.

**11–14일:** ID와 OOD validation에서 같은 `fmax`의 성공률 대 wall-time 및 평가 횟수 곡선을 비교한다. PDD가 조정된 FIRE/LBFGS 및 direct endpoint보다 낫지 않거나 head 복제 비용이 너무 높으면 그 결과를 기록하고 MD와 별개 연구 방향으로 유지한다. 기초 비교에서 우월한 영역이 보이면 OC20 전체 train과 OOD 체계, OC20-Dense/DFT 후속 평가로 확장한다. DDP는 단일 GPU 병목·모델 구조를 확인한 뒤 구현한다.

## 연구 포지셔닝

문헌상 MLIP를 이용한 relaxation 가속 자체는 새롭지 않다. [AdsorbML](https://www.nature.com/articles/s41524-023-01121-5)은 MLIP로 후보를 relax한 뒤 DFT 단일점/재최적화를 결합했고, [AdsorbDiff](https://proceedings.mlr.press/v235/kolluru24a.html)와 [AdsorbFlow](https://arxiv.org/abs/2602.19289)는 낮은 에너지 adsorbate 배치를 생성한다. 따라서 새로운 주장은 **공개 OC20 DFT relaxation 경로로 pretrained potential의 여러 구조 변화를 한 번의 shared backbone 평가에서 예측하고, MLIP의 최종 수렴 품질을 보존하면서 총 relaxation 시간을 단축한다**는 것이다. 물리적 수렴과 벽시계 시간에서 경쟁 기준을 이겨야 한다. 이 프로젝트의 기존 MD 코드는 데이터 로더·위상공간 head·교사용 적분·PBC 가정이 달라 relaxation에 그대로 적용할 수 없고, 공통 backbone/다중 head 아이디어만 재사용한다.

## 실행된 첫 PoC

[OC20 `*H` PoC 결과](../results/relaxation_oc20_poc_2026-09-28.md)는 공식 궤적 1,000개를 사용해 DFT 경로의 4-step 내부 좌표를 다중 head가 예측할 수 있는지 시험한다. 먼저 저장된 **DFT force**를 MLIP force의 대용 입력으로 학습했고, 뒤이어 공개 GemNet-OC-2M force로 8개 계의 실제 MLIP refinement를 실행했다. 처음에는 FIRE를 기준으로 썼으나, LBFGS로 다시 측정한 한 블록 실험에서 GemNet 호출이 기준 234회, PDD+LBFGS 207회, 직접 endpoint+LBFGS 213회였다. 두 블록에서는 각각 234/198/220회였고, 모든 방법이 8/8 수렴했다. 이 작은 CPU 표본만으로 견고한 가속이나 다중 head의 독자적인 이득을 입증할 수 없다. 학생은 작은 새 backbone이며 원 PDD의 student-state on-policy target과 pretrained GemNet head 복제는 아직 적용하지 않았다.

[별도 100계 LBFGS 평가](../results/relaxation_oc20_holdout_2026-09-30.md)에서는 기준이 100/100 수렴·4,377회 GemNet 호출, PDD 두 블록이 98/100 수렴·4,302회, 직접 endpoint 두 번이 98/100 수렴·4,123회였다. PDD의 총 호출 절감은 1.7%에 그쳤고 계 단위 bootstrap 구간은 0을 포함했다. PDD 수렴 계 중 10계는 기준보다 최종 에너지가 0.05 eV 넘게 높았고 7계는 DFT 흡착 위치 오차가 0.2 Å 넘게 증가했다. **현재 구현으로 품질을 유지한 relaxation 가속을 주장할 수 없다.** 이 후속 평가는 아직 `*H` 아카이브의 작은 validation subset이며 공식 OC20 ID/OOD 검증은 아니다.

같은 100계에서 사후 탐색한 **한 블록** PDD는 99/100 수렴·4,251회 호출이었다. 에너지 +0.05 eV 초과 계는 7개, 흡착 위치 오차 +0.2 Å 초과 계는 7개였다. 두 번째 제안을 생략하면 일부 실패는 완화됐으나, 첫 제안에서 다른 구조로 이동하는 사례와 직접 endpoint 대비 불분명한 다중 head 이득은 남았다. 기준 wall time은 앞선 실행에서 재사용했으므로 한 블록의 wall time 가속 배수는 보고하지 않는다.
