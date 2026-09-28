# 서버 실험 가이드: PDD로 eSEN MD 가속

기준: 2026-09-28. 이 문서가 서버 작업의 시작점이다. `README.md`는 코드 사용법, `results/`는 지금까지의 실측 기록이다. 논문의 PDD를 분자 동역학에 적용한다는 *연구 가설*과 이미 확인된 결과를 구별해서 보고한다.

## 0. 지금 클론하면 어디까지 가능한가

| 항목 | 현 상태 |
|---|---|
| AD-3 다운로드, eSEN 모델 로드, force warm start, PDD 학습, 체크포인트 재개 | 구현됨; 단일 GPU로 사용 가능 |
| 동일 eSEN 포텐셜의 에너지 미분 teacher와 PDD 비교 | 구현됨; T4에서 짧은 실험 확인 |
| head별 고정 검증, rollout 진단, coarse Verlet·direct baseline, 추론 시간 측정 | 구현됨 |
| 4 GPU 동기식 학습, 자동 실험 추적, 독립 validation/test 분리 | **미구현** |
| 안정적인 장시간 MD 또는 평형 샘플링 가속 | **미입증** |

따라서 서버 에이전트는 지금 바로 **1 GPU 재현과 4개 GPU에 서로 다른 독립 실험 배치**를 시작할 수 있다. 현재 CLI에 `torchrun --nproc_per_node=4 pdd-md ...`를 쓰면 안 된다. 코드에는 process group, rank별 샘플링, DDP gradient synchronization이 없다. 4 GPU를 한 모델의 동기식 학습에 쓰려면 아래 5단계의 구현과 검증이 먼저 필요하다.

`scripts/run_experiment.sh`는 기본 OpenMM teacher 실험이다. **최근의 주 실험인 eSEN energy-gradient 동일 포텐셜 설정이 아니다.** 이 가이드의 명령을 주 실험에 사용한다. README의 초기 Colab 및 CPU 실험 결과도 서로 다른 teacher 또는 짧은 학습 예산을 포함하므로 숫자를 섞어 비교하지 않는다.

## 1. 과학적 질문과 성공 조건

고정된 AD-3 alanine dipeptide 구조·속도에서 시작하는 결정론적 NVE teacher를 정의한다. teacher는 `eSEN-sm-direct` 체크포인트의 **energy head를 좌표로 미분한 force**로 0.5 fs velocity Verlet 적분한다. 학생은 같은 체크포인트의 shared eSEN backbone과 복제한 direct-force head들로 한 번의 backbone 평가에서 L개의 내부 시간 간격 평균 위상공간 속도를 예측한다. head별 목표는 학생이 만들어낸 내부 상태에서 teacher가 한 fine step 진행했을 때의 평균 변화량이다. AD-3 파일의 *원래 Langevin 경로*를 재현하는 실험은 아니다.

검증할 주장은 세 층으로 나눈다.

1. **학습:** 별도 validation 시작 상태에서 각 head의 평균 속도·가속도 오차가 감소하는가?
2. **동역학:** 반복 적용한 위치·속도·에너지·구조 및 분포 지표가 허용 범위에 남는가?
3. **가속:** 동일 eSEN energy 포텐셜, 동일 하드웨어, 동일 simulated time, 같은 batch와 정확도 조건에서 PDD가 fine/coarse Verlet와 direct transition보다 빠른가?

숫자 하나로 성공을 주장하지 않는다. 특히 backbone 호출 수나 짧은 구간 속도만으로 MD 가속이라고 하지 않는다. 최종 표는 시간 대비 정확도·안정성의 Pareto 비교다. 위치 RMSE 0.1 Å는 현재 코드의 디버그 경계일 뿐 물리적 타당성 기준이 아니다. 평형 샘플링 주장을 하려면 별도 Langevin/thermostat 프로토콜과 장시간 통계 평가가 필요하다.

## 2. 서버 준비와 데이터

최소 요구: Linux, Python 3.11 또는 3.12, CUDA가 동작하는 PyTorch, NVIDIA 드라이버, 충분한 로컬 디스크, Hugging Face의 `facebook/OMol25` 체크포인트 접근 권한. 3090 4개의 실제 가용 메모리·드라이버·CUDA 버전은 서버에서 확인한다. `fairchem-core>=2.23,<2.24`가 현재 코드의 로더 및 head 구조와 맞춰진 범위다. 새 버전으로 올릴 때는 모델 출력과 force 정규화의 동일성부터 재검증한다.

```bash
git clone https://github.com/minkyu1022/PDD_MD_acceleration.git
cd PDD_MD_acceleration
git rev-parse HEAD
nvidia-smi
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 서버 CUDA에 맞는 PyTorch 설치를 먼저 확인한 후:
python -m pip install -e '.[esen,test]'
python -m pytest -q
pdd-md download-data --data-root data
pdd-md inspect-data --data-root data --split train
```

공식 [Timewarp AD-3 dataset](https://huggingface.co/datasets/microsoft/timewarp/tree/main/AD-3) 페이지는 AD-3 폴더를 약 899 MB로 표시한다. 코드에서 확인한 파일은 train 800,000·test 400,000 저장 frame으로 이루어진 **각각 하나의 긴 궤적**이다. `download-data`가 train/test NPZ와 topology PDB를 지정된 디렉터리에 놓는다. 각 저장 frame을 새 시작 상태로 샘플링할 수 있지만 인접 frame은 강하게 상관될 수 있다. 120만 개의 독립 궤적이라고 해석하지 않는다. 저장 frame 간격은 불규칙하므로 NPZ의 연속 frame을 0.5 fs teacher target으로 취급하지 않는다. 파일 검증: 크기, SHA-256, frame 수, atom 수, `step/time`, finite 값, topology 일치를 manifest에 기록한다. 코드는 원래 AD-3 force와 새 eSEN teacher force의 차이도 보고한다. 두 포텐셜의 레이블을 혼합하지 않는다.

[OMol25 체크포인트 카드](https://huggingface.co/facebook/OMol25)에서 모델 라이선스를 승인하고 서버 계정에서 `hf auth login`을 완료한다. 접근 권한은 Colab/Mac에서 서버로 자동 전파되지 않는다. repo에는 토큰이나 체크포인트를 넣지 않는다. `esen-sm-direct-all-omol` 모델 ID로 FAIR-Chem이 다운로드하게 하거나 승인된 파일 `checkpoints/esen_sm_direct_all.pt`를 로컬에 보관한다. 아래 명령에서는 로컬 파일을 사용한다. checkpoint SHA-256 및 허가·버전을 run manifest에 기록한다. 데이터나 모델을 못 내려받으면 에이전트는 오류와 필요한 접근만 보고하고 다른 포텐셜로 조용히 바꾸지 않는다.

```bash
mkdir -p checkpoints runs
hf auth whoami
# 승인된 계정에서; 파일은 checkpoints/esen_sm_direct_all.pt에 저장된다.
hf download facebook/OMol25 checkpoints/esen_sm_direct_all.pt --local-dir .
sha256sum checkpoints/esen_sm_direct_all.pt data/AD-3/train/*.npz data/AD-3/test/*.npz
```

`hf download`의 파일 경로와 `--local-dir` 동작은 [Hugging Face CLI 문서](https://huggingface.co/docs/huggingface_hub/guides/cli)에 따른다. 정확한 repository revision도 manifest에 남겨 같은 파일을 다시 받을 수 있게 한다.

## 3. 데이터 분할과 재현성: 큰 실험 전 필수 수정

현재 `validate-pdd`, `evaluate`, `diagnose`, `benchmark`는 모두 AD-3의 **test trajectory**를 사용한다. Colab에서 반복해서 본 숫자는 탐색용이다. 향후 하이퍼파라미터 선택에 같은 test를 계속 쓰면 최종 시험이 오염된다. 서버의 첫 코드 작업은 split 파일을 도입하는 것이다.

- train AD-3 궤적을 시간상 연속 구간으로 나눠 train과 validation을 만든다. 블록 주변에 충분한 시간 buffer를 두고 겹치는 시작 상태를 피한다. validation의 시작 index는 저장한다. 구간 길이·buffer는 궤적 자기상관을 확인한 후 확정한다.
- AD-3 test 궤적은 최종 평가까지 잠근다. 하이퍼파라미터·조기 종료·checkpoint 선택에 쓰지 않는다. 이미 test를 보며 만든 PoC 설계는 탐색 결과로 명시한다.
- 한 궤적만의 split로는 분자·포텐셜·온도 일반화가 증명되지 않는다. 후속 시스템은 별도 dataset 또는 별도 생성 궤적과 topology를 요구한다.
- 모든 run에 Git SHA, data/checkpoint SHA, PyTorch·CUDA·FAIR-Chem 버전, seed, precision, GPU 모델, split index, teacher 정의, dt, block size, batch size, optimizer와 설정을 저장한다. 결과 JSON/체크포인트는 Git에 넣지 않고 서버의 지속 저장소에 둔다. 코드 커밋과 작은 요약 표만 Git에 남긴다.

## 4. 단일 GPU 기준선: 그대로 실행 가능한 경로

각 명령의 `--data-root`, checkpoint 경로, `--max-frames`, seed를 동일하게 유지한다. `--teacher-backend esen-energy`가 **첫 force run에서 필수**다. 이를 생략하면 기본값은 OpenMM으로 바뀌고 후속 checkpoint가 다른 teacher를 상속한다. 모든 단계에서 checkpoint 메타데이터의 `teacher_backend`, `teacher_checkpoint`, `dt_ps`, topology를 확인한다. 아래 2,048-frame·100/1,000-update 예제는 Colab 재현용 작은 예산이며 본 실험의 최종 학습량이 아니다.

```bash
export CUDA_VISIBLE_DEVICES=0
pdd-md train-force --backend esen \
  --checkpoint checkpoints/esen_sm_direct_all.pt \
  --teacher-backend esen-energy --data-root data \
  --output runs/repro/force_100.pt --max-frames 2048 \
  --steps 100 --batch-size 4 --device cuda --log-every 20
pdd-md evaluate-force --data-root data \
  --force-checkpoint runs/repro/force_100.pt \
  --output runs/repro/force_eval.json --max-frames 2048 \
  --samples 64 --device cuda
pdd-md train-pdd --data-root data \
  --force-checkpoint runs/repro/force_100.pt \
  --output runs/repro/pdd_L4_B16_1000.pt --max-frames 2048 \
  --steps 1000 --batch-size 16 --max-block 4 --block-sizes 4 \
  --prefix-blocks 2 --learning-rate 2e-5 --save-every 250 \
  --log-every 100 --device cuda
pdd-md validate-pdd --data-root data \
  --pdd-checkpoint runs/repro/pdd_L4_B16_1000.pt \
  --output runs/repro/val_exploratory.json --max-frames 2048 \
  --samples 64 --batch-size 16 --block 4 --device cuda
pdd-md evaluate --data-root data \
  --pdd-checkpoint runs/repro/pdd_L4_B16_1000.pt \
  --output runs/repro/eval_8.json --max-frames 2048 \
  --samples 8 --fine-steps 8 --blocks 4 --device cuda
pdd-md diagnose --data-root data \
  --pdd-checkpoint runs/repro/pdd_L4_B16_1000.pt \
  --output runs/repro/diagnose_80.json --max-frames 2048 \
  --samples 8 --horizons 4 8 20 40 80 --block 4 --device cuda
```

현재 CLI의 validation/evaluation 명령은 test를 읽으므로 위 재현 결과는 **exploratory**로 명명했다. 3단계 분할 수정 후 본 실험에는 validation split을 지정한다. `--resume OLD --steps N`의 N은 *추가 횟수*가 아니라 총 update 횟수다. eSEN 모델의 force warm start는 현재 teacher force 레이블을 샘플마다 직렬 계산한다. 길거나 큰 batch에서 이 부분의 시간을 따로 계측하고 batched teacher로 개선한다. 전체 AD-3 800,000 frame을 RAM에 읽는 현재 loader의 메모리도 먼저 측정한다.

### 기존 T4 실측으로 기대할 것

[동일 포텐셜 실험 기록](../results/same_potential_2026-09-26.md)과 [batch-16 학습 곡선](../results/l4_batch16_learning_curve_2026-09-26.md)을 재현 기준으로 삼는다. T4 L4/batch16 1,000 update에서 8-step 위치 RMSE는 약 0.00887 Å, 평균 절대 에너지 드리프트는 약 11.21 kJ/mol이었다. 40-step에는 네 시험 시작점 모두 0.1 Å 미만, 80-step에는 한 궤적 비정상 및 나머지도 0.1 Å 초과였다. 이는 좋은 최종 수치가 아니라 파이프라인 sanity check다. 서버 하드웨어에서는 wall time 자체를 T4와 비교하지 않는다.

## 5. 4×3090 확장 순서

**즉시:** GPU 0에서 위 재현, GPU 1–3에서 독립 실험을 각각 단일 GPU 프로세스로 실행한다. 예: GPU 1은 같은 L4의 batch 32/64 메모리·학습률 검사, GPU 2는 L2/L4 curriculum, GPU 3은 direct-transition baseline 및 동일 포텐셜 coarse Verlet 평가. 독립 run마다 다른 출력 디렉터리를 사용하고 validation split 완성 전에는 과학적 승자를 고르지 않는다. 각 GPU 프로세스 앞에 `CUDA_VISIBLE_DEVICES=0`, `=1`, `=2`, `=3`을 따로 붙이고 프로세스 내부의 `--device cuda`를 사용한다. 예를 들어 force checkpoint가 준비된 뒤 GPU 1에서 `CUDA_VISIBLE_DEVICES=1 pdd-md train-pdd --data-root data --force-checkpoint runs/repro/force_100.pt --output runs/batch32/pdd.pt --max-frames 2048 --steps 1000 --batch-size 32 --max-block 4 --block-sizes 4 --prefix-blocks 2 --device cuda`로 독립 run을 시작할 수 있다. 4 GPU 전체를 한꺼번에 쓰기 전에 1 GPU에서 배치 16→32→64를 점진적으로 올려 최대 메모리, step time, 샘플/초와 loss를 기록한다. OOM이면 batch를 낮춘다. 더 큰 batch는 learning rate와 최적 update 수를 자동으로 보장하지 않는다.

**다음:** 단일 모델 DDP를 구현할 때 `torchrun`으로 rank마다 GPU 하나를 쓰고, 학생만 DDP로 감싼다. teacher는 각 rank의 장치에 읽기 전용으로 두며, rank별로 겹치지 않는 train 시작 상태·독립 RNG를 샘플링한다. `global_batch = per_gpu_batch × 4 × accumulation_steps`와 유효 teacher target 수를 로그에 남긴다. rank 0만 원자적 체크포인트와 요약을 쓰고, 재개 시 optimizer, scheduler, scaler, *각 rank*의 RNG·sampler state를 복원한다. 최초에는 FP32의 1-GPU/4-GPU 동등성 및 single-step gradient parity를 확인한다. gradient accumulation은 실제 GPU 메모리나 teacher 연산량을 줄이지 않는다. AMP와 `torch.compile`은 성능 측정 후 각각 별도 ablation으로 추가하고, 에너지 미분 force와 rollout 안정성이 같은지 검증한다. PyTorch의 [DDP 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)는 GPU별 프로세스와 명시적 데이터 샤딩을 요구한다.

**GPU 배치보다 우선 측정할 병목:** teacher energy-gradient 두 번, eSEN 그래프 생성, student forward/backward, CPU↔GPU 전송, 검증 및 저장 시간. `teacher.force_calls`는 현재 배치 대상 상태 수에 비례한 논리적 호출 수로 기록된다. batched GPU 커널 호출 수와 다르므로 실제 타이밍을 별도 기록한다. 프로파일러는 [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)를 사용한다. 4-GPU 속도 향상은 동일 **global batch와 동일 target 수**에서의 학습 처리량 및 time-to-quality로 평가한다. 단순히 update/초를 비교하지 않는다.

## 6. 단계별 실험 계획과 진행 조건

| 단계 | 실험 | 다음 단계 조건 |
|---|---|---|
| A. 환경·데이터 | 설치, 데이터·모델 hash, force/energy 단위, topology, 두 teacher force 평가의 결정성, 재개 일치 | smoke 통과; finite force; teacher/checkpoint/단위 일치 |
| B. 재현 | 2,048 train 시작점과 기존 seed로 T4 PoC 재현; GPU timing/메모리 측정 | fixed-set head 오차·짧은 rollout이 기존 결과와 같은 대략적 수준; 차이는 조사 |
| C. split·계측 | train/validation 시간 블록 분리, manifest, JSONL 로그, long-rollout 평가 자동화 | test를 사용하지 않고 checkpoint 선택 가능 |
| D. 단일 GPU scale | 전체 train pool로 1k→5k→20k update, L=1/2/4와 batch16/32/64, prefix depth·loss weight 비교 | validation head와 8/40/80-step 품질이 함께 개선; 비정상 궤적 비율 감소 |
| E. baseline | 동일 teacher의 fine Verlet, coarse Verlet L=2/4/8, 직접 전이 모델을 동일 시작점·시간으로 평가 | PDD의 정확도–속도 곡선이 baseline보다 우수한 영역 확인 |
| F. 4-GPU | 독립 sweep 우선; 병목에 따라 DDP 구현 및 동등성 확인 | 동일 target 수의 time-to-quality 개선, 재개·재현 통과 |
| G. 장시간/통계 | 긴 NVE rollout과 여러 독립 seed·초기상태, 구조 붕괴·에너지·Ramachandran 분포; 필요하면 별도 thermostatted teacher | 목표 물리량과 허용오차를 사전에 정하고, 기존 baseline 대비 통계·wall time 비교 |

초기 L=4 고정부터 시작하는 이유는 현 PoC의 비교 자료가 있기 때문이다. L=8, 더 긴 horizon, 다른 분자 및 다양한 포텐셜은 L4의 안정성 원인을 파악한 뒤 연다. 1,500 update 실험에서 손실 가중치를 바꿔 head 속도 오차는 줄었지만 장시간 정확도와 에너지 보존은 해결되지 않았다. 단순 update 증가나 scalar loss 최소값으로 모델을 선택하지 않는다. 잘못된 포텐셜·비정상 상태·exploding error가 나오면 더 오래 학습하기 전에 원인을 진단한다.

## 7. 저장할 로그와 결과 표

**매 학습 로그 간격:** update, wall time, 실행 seed/GPU, global batch, L 및 선택 head 분포, prefix 길이, teacher target 평가 수, 실제 teacher·student 시간, samples/second, GPU peak allocated/reserved memory, learning rate, grad norm·clip 비율, 총 loss와 mean-velocity/acceleration 항을 각각 기록한다. NaN/Inf, force/속도/가속도의 극단값도 카운트한다. 현재 CLI는 일부만 기록하므로 서버 agent가 확장해야 한다.

**고정 validation checkpoint마다:** head별 qdot/accel RMSE(물리 단위), head별/평균 scaled loss와 사용 scale, teacher force RMSE, one-block 상태 오차, 8/40/80/더 긴 fine-step rollout의 q/v endpoint·path error 분포(중앙값, p90, 최대값), finite 비율, 0.1 Å crossing time(디버그용), 총 에너지 drift의 절대값뿐 아니라 signed drift와 시간에 따른 slope, 최고 force/최소 원자 간 거리·bond-length 이상률. 장시간 통계에는 φ/ψ의 주기 경계를 고려한 Ramachandran histogram/free-energy surface, basin 점유율·전이율·자기상관 및 신뢰구간을 teacher와 비교한다. 현재 구현되지 않은 지표는 구현 후 사용한다.

**속도:** 같은 GPU에서 warmup 후 CUDA synchronize, 동일 batch·simulated time·시작점으로 p50/p90 wall seconds, ns/day 또는 simulated fs/s, trajectories/s, step당 backbone·energy-gradient 평가 횟수, 총 추론 비용 및 처리 실패 비율을 기록한다. teacher force 계산·graph 구축·head·적분 포함 여부를 명시한다. 모델 로딩·초기 전송은 online rollout과 별도로 기록한다. speedup은 해당 정확도·안정성 조건을 통과한 모델에 한해 표시한다. teacher potential이 다른 결과를 speedup이라 부르지 않는다. `benchmark-mlip`은 현재 compute-only 진단이고 physics 조건을 보증하지 않는다.

**실험 단위:** 적어도 몇 개의 seed와 여러 떨어진 시작 시점을 사용해 불확실성을 표시한다. 한 길고 상관된 궤적에서 뽑은 수천 frame을 독립 표본처럼 신뢰구간에 사용하지 않는다. 최종 test 평가는 결정한 설정에 한 번 수행하고, 실패까지 포함한 원시 결과를 보관한다.

## 8. 코드·논문 참고자료

| 자료 | 용도 |
|---|---|
| [PDD 원 논문](https://arxiv.org/html/2607.26004), [프로젝트 페이지](https://research.nvidia.com/labs/genair/pdd/) | mean-velocity, 학생 내부 상태에서의 head별 teacher 감독, variable block, head fusion. **원 논문은 이미지/비디오 모델이지 MD 물리 검증이 아니다.** 프로젝트 페이지는 현재 공식 PDD 코드가 “coming soon”이라고 표시한다. |
| [NVlabs/FastGen](https://github.com/NVlabs/FastGen) | 관련 공식 학습 인프라 참고. 현재 공개 README에서 PDD 구현을 확인하지 못했으므로 PDD ground truth 코드라고 취급하지 않는다. |
| [Timewarp 논문](https://arxiv.org/abs/2302.01170), [코드](https://github.com/microsoft/timewarp), [AD-3 파일](https://huggingface.co/datasets/microsoft/timewarp/tree/main/AD-3) | prior work, dataset와 원래 force field·Langevin 조건. Timewarp는 평형 분포를 대상으로 MCMC 보정을 사용한다; 본 결정론적 PDD와 성능 주장이 다르다. |
| [OMol25 논문](https://arxiv.org/abs/2505.08762), [체크포인트 카드](https://huggingface.co/facebook/OMol25), [FAIR-Chem](https://github.com/facebookresearch/fairchem) | eSEN pretrained backbone, energy/direct-force heads, 체크포인트 라이선스와 로더. |
| 이 repo의 `src/pdd_md/{teacher,model,train,diagnostics,evaluate}.py` | 현재 구현과 결과 JSON의 정확한 정의. 코드 변경 시 여기의 단위·teacher 계승·head fusion을 확인. |

## 9. 서버 에이전트에게 전달할 작업 문구

> 이 저장소의 `docs/SERVER_EXPERIMENT_GUIDE.md`를 먼저 읽고, `results/same_potential_2026-09-26.md`와 `results/l4_batch16_learning_curve_2026-09-26.md`를 확인해 주세요. 4×3090 서버에서 `eSEN energy-gradient velocity-Verlet` teacher와 같은 포텐셜의 PDD 실험을 이어갑니다. 우선 환경·데이터·체크포인트를 검증하고 1 GPU 재현을 완료하세요. 다음으로 AD-3 train 궤적 내부에 시간 블록 validation split을 만들고 test를 잠근 뒤, 로그와 장시간 안정성 지표를 추가하세요. 4 GPU는 독립 실험에 먼저 배분하고, DDP를 구현·검증하기 전에는 `torchrun`을 쓰지 마세요. 같은 potential의 fine/coarse Verlet 및 direct baseline과 동일한 시간·시작점·하드웨어에서 비교하세요. 각 단계에서 명령, Git SHA, 설정, dataset/model hash, checkpoint 경로, 검증 수치, 실패를 기록하고 재개 가능한 상태로 커밋하세요. checkpoint와 데이터는 Git에 올리지 마세요. 접근 권한이나 모델 파일이 없으면 무엇이 필요한지 정확히 보고하세요.
