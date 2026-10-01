# T02 — 결과 기록(JSON) 추가

정본 `CLAUDE.md`. manual mode, 멈춤 지점 두 곳(§2).

## 목표 (검증할 주장)
`--out_dir`를 주면 에폭별 정확도와 test·valid 에피소드별 기록이 파일로 남고, 주지 않으면 원본과 같은 출력이 나온다. 기록 기능은 학습·평가 수치를 바꾸지 않는다.

## 바꿀 것 (**멈춤 지점 1: 수정 전 계획 승인**)
- `argument.py`: `--out_dir` (str, 기본 `None`) 추가.
- `model.py`:
  - `train_epoch`에서 mode가 `valid`·`test`일 때 에피소드마다 `{epoch, mode, ep_idx, classes(class_selected), acc, f1}`를 리스트에 쌓는다. 계산 경로·난수 호출은 그대로 둔다.
  - `train`에서 에폭마다 `{epoch, train_acc, train_f1, valid_acc, valid_f1, test_acc, test_f1}`를 쌓는다.
  - `out_dir`가 주어지면 학습 종료 시 다음 두 파일을 쓴다.
    - `run.json`: `config`(args 전체), `epochs`(위 에폭 기록), `final`(train()이 반환하는 11개 값, 키 이름 포함), `wall_time_sec`
    - `episodes.jsonl`: valid·test 에피소드 기록 한 줄에 하나
  - `out_dir`가 `None`이면 파일을 쓰지 않는다.
- 텐서 값은 파이썬 float로 바꿔 저장한다.
- 원본 출력(tqdm.write, print)은 그대로 둔다.

## 확인 실행
- A: `CUDA_VISIBLE_DEVICES=0 python main.py --dataset Amazon_clothing --way 5 --shot 1 --seed 0 --num_seed 1 --device 0 --out_dir results/T02/amac_5w1s_s0`
- B: 같은 명령에서 `--out_dir` 없이 실행 (`results/T02/amac_5w1s_s0_noout.log`)

## 완료 조건
1. A의 `run.json` 에폭별 정확도 = A의 콘솔 출력 값
2. 에폭마다 `test_acc` = 그 에폭 test 에피소드 50개 `acc`의 평균 (차이 < 1e-6)
3. A와 B의 콘솔 에폭별 정확도 비교: T01에서 결정적이었다면 완전히 같아야 한다. 비결정적이었다면 차이를 표로 보고한다.
4. A와 T01 4단계 Amazon_clothing 로그의 비교(같은 판정 기준)

## 보고 (**멈춤 지점 2**) — `reports/T02_logging.md`
- diff 요약(파일·함수·줄 수), `run.json` 키 구조 예시(값 일부), `episodes.jsonl` 첫 2줄
- 완료 조건 1~4 표
- 이상 징후, 커밋 해시, 태그 `t02`

## 하지 말 것
CLAUDE.md §7 전부. 계산 경로 변경. 난수 호출 추가. 기록 대상 외 지표 추가.
