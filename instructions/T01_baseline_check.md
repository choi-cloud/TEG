# T01 — 원본 TEG 기준 실행: 데이터·GPU 지정·결정성·소요 시간

정본 `CLAUDE.md`. manual mode, 멈춤 지점 두 곳(§2).

## 목표 (검증할 주장)
원본 TEG가 세 데이터셋에서 끝까지 돌고, `CUDA_VISIBLE_DEVICES=<물리> --device 0` 형식으로 지정한 물리 GPU에서 실행되며, 같은 seed 재실행의 수치가 같은지와 run 1회 소요 시간을 안다.

## 0단계 — 문서 반영 (코드 변경 없음)
- 첨부된 `CLAUDE.md`를 리포 루트에, `instructions/T01~T04*.md`를 `instructions/`에, `relation_textbook_methodology_v0.md`를 `docs/`에 둔다.
- `.gitignore`에 `results/`를 추가한다.
- 커밋: `docs: add CLAUDE.md, instructions T01-T04, methodology v0`.

## 1단계 — 데이터 확인
- 다음 9개 파일의 존재와 크기를 표로 보고한다: `dataset/{Amazon_clothing, Amazon_electronics, dblp}/{이름}_network`, `{이름}_train.mat`, `{이름}_test.mat`.
- 하나라도 없으면 여기서 멈추고 보고한다.

## 2단계 — GPU 지정 확인 (수정 전)
- 실행: `CUDA_VISIBLE_DEVICES=2 python main.py --dataset Amazon_clothing --way 5 --shot 1 --epochs 1 --seed 0 --device 0`
- 실행 중 `nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv`로 프로세스가 올라간 물리 GPU를 기록한다(물리 2번의 bus id와 대조).
- 배경: `embedder.py`가 프로세스 안에서 `os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)`로 덮어쓴다. 외부 지정이 무시될 수 있다(추정).

## 3단계 — 필요할 때만 수정 (**멈춤 지점 1: 수정 전 계획 승인**)
- 2단계에서 물리 2번이 아니면: `embedder.py`의 위 한 줄을 삭제한다. 나머지(`self.device = f'cuda:{args.device}'`, `torch.cuda.set_device`)는 그대로 둔다.
- 수정 후 2단계를 다시 실행해 물리 2번에 올라가는지 확인한다.
- 이 변경은 수치 계산에 관여하지 않는다. 커밋: `fix: respect external CUDA_VISIBLE_DEVICES`.
- 2단계에서 이미 물리 2번이면 수정하지 않고 그 사실을 보고한다.

## 4단계 — 기준 실행 (3 run, 병렬)
- `results/T01/<dataset>_5w1s_seed0.log`로 출력을 저장한다(`2>&1 | tee`).
- `CUDA_VISIBLE_DEVICES=0 python main.py --dataset Amazon_clothing --way 5 --shot 1 --seed 0 --num_seed 1 --device 0`
- `CUDA_VISIBLE_DEVICES=1 ... --dataset Amazon_electronics ...`, `CUDA_VISIBLE_DEVICES=2 ... --dataset dblp ...` (나머지 인자 동일, epochs 기본값 10)
- 각 run의 시작~종료 시각(초)을 기록한다. 구조 특징 생성("Generating a structural feature..." ~ "Done.")에 걸린 시간도 따로 기록한다.

## 5단계 — 결정성
- Amazon_clothing 5w1s seed 0을 같은 GPU(물리 0)에서 한 번 더 실행해 `results/T01/Amazon_clothing_5w1s_seed0_rep.log`에 저장한다.
- 4단계 로그와 에폭별 `acc_train`, `acc_valid`, `acc_test`를 비교해 전부 같은지, 다르면 첫 번째로 달라진 에폭과 값을 보고한다.

## 완료 조건
1. 데이터 9개 파일 존재
2. `CUDA_VISIBLE_DEVICES=2 ... --device 0` 실행이 물리 GPU 2번에서 돈다
3. 세 데이터셋 run이 오류 없이 끝나고 `Acc_Test_At_Best_Valid` 출력
4. 결정성 비교 결과(같음/다름)와 run별 소요 시간 보고

## 보고 (**멈춤 지점 2**) — `reports/T01_baseline_check.md`
- 표 1: 데이터 파일 존재·크기
- 표 2: GPU 지정 결과(수정 전/후, bus id)
- 표 3: 데이터셋별 에폭 0~10의 train/valid/test 정확도, `Acc_Test_At_Best_Valid`와 그 에폭, 소요 시간(전체/구조 특징)
- 표 4: 결정성 비교
- 이상 징후
- 커밋 해시, 태그 `t01`

## 하지 말 것
CLAUDE.md §7 전부. 위 한 줄 외의 코드 수정. 지정 외 설정 실행.
