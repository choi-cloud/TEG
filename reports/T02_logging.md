# T02 보고 — 결과 기록(JSON) 추가 (2026-10-01)

## 1. 변경 파일과 위치
커밋: 이 보고서와 같은 커밋 (태그 `t02`). 삭제 줄 0(CLAUDE.md 1줄 교체 제외).

| 파일 | 위치 | 추가 줄 | 내용 |
|---|---|---|---|
| `argument.py` | `parse_args` | +1 | `--out_dir` (str, 기본 `None`) |
| `argument.py` | `config2string` 제외 목록 | +1 | `"out_dir"` 추가 (이상 징후 1) |
| `model.py` | import | +3 | `os`, `json`, `time` |
| `model.py` | `train_epoch` (정확도 append 뒤) | +12 | mode가 valid·test면 `{epoch, mode, ep_idx, classes, acc, f1}`를 `self.episode_records`에 추가 |
| `model.py` | `train` (루프 전) | +4 | `start_time`, `self.epoch_records`, `self.episode_records` 초기화 |
| `model.py` | `train` (test 평가 뒤) | +12 | 에폭 기록 `{epoch, train/valid/test _acc/_f1}` |
| `model.py` | `train` (return 전) | +26 | `out_dir`가 있을 때만 `run.json`, `episodes.jsonl` 작성 |
| `CLAUDE.md` | §4 | +10 −1 | 실행 형식을 `CUDA_VISIBLE_DEVICES=<GPU UUID>`로 변경, UUID 표 |
| `CLAUDE.md` | §7 | +1 | 재시도 로그 `_retry1` 규칙 |

`run.json` 키 구조 (A, 값 일부):
```json
{
  "config": {"lr": 0.001, "epochs": 10, "device": 0, "seed": 0, "dataset": "Amazon_clothing", "...": "...", "out_dir": "results/T02/amac_5w1s_s0"},
  "epochs": [{"epoch": 0, "train_acc": 0.6376000000000002, "train_f1": 0.6092232316702905, "valid_acc": 0.5519999999999999, "valid_f1": 0.5164046972635207, "test_acc": 0.6248000000000001, "test_f1": "..."}, "... 11개"],
  "final": {"best_acc_train": 0.8816000000000003, "best_f1_train": 0.8741152181152181, "best_epoch_train": 10, "best_acc_valid": 0.7960000000000003, "best_f1_valid": 0.7861651015651014, "best_epoch_valid": 7, "best_acc_test": 0.8360000000000004, "best_f1_test": 0.8301920856920856, "best_epoch_test": 6, "test_acc_at_best_valid": 0.8048000000000003, "test_f1_at_best_valid": 0.798966477966478},
  "wall_time_sec": 13.513296365737915
}
```

`episodes.jsonl` 첫 2줄 (총 1100줄 = 11 에폭 × (valid 50 + test 50)):
```
{"epoch": 0, "mode": "valid", "ep_idx": 0, "classes": [3.0, 24.0, 44.0, 34.0, 35.0], "acc": 0.56, "f1": 0.49712121212121213}
{"epoch": 0, "mode": "valid", "ep_idx": 1, "classes": [44.0, 35.0, 43.0, 74.0, 68.0], "acc": 0.44, "f1": 0.3664335664335664}
```

## 2. 실행 명령과 출력
GPU: 물리 0 (`GPU-4b1a9f61-18d5-93a0-790e-87ba97de881e`). A → B 순차 실행.
- A: `CUDA_VISIBLE_DEVICES=GPU-4b1a9f61-… python main.py --dataset Amazon_clothing --way 5 --shot 1 --seed 0 --num_seed 1 --device 0 --out_dir results/T02/amac_5w1s_s0 2>&1 | tee results/T02/amac_5w1s_s0.log`
- B: 같은 명령, `--out_dir` 없음 → `results/T02/amac_5w1s_s0_noout.log`

| run | exit | Acc_Test_At_Best_Valid | F1 | 에폭 | 생성 파일 |
|---|---|---|---|---|---|
| A | 0 | 0.8048000000000003 | 0.798966477966478 | 7 | `run.json`, `episodes.jsonl` |
| B | 0 | 0.8048000000000003 | 0.798966477966478 | 7 | 없음 |
| T01 4단계 | 0 | 0.8048000000000003 | 0.798966477966478 | 7 | — |

## 3. 완료 조건 대비 (조건 3·4는 완전 일치 기준)
| 조건 | 결과 | 충족 |
|---|---|---|
| 1. A `run.json` 에폭별 정확도 = A 콘솔 | 11 에폭 × train/valid/test 33개, 소수 4자리 문자열 불일치 0. `final` 3값(acc, F1, 에폭)이 콘솔 `==>` 줄 전체 자릿수와 같음 | 충족 |
| 2. 에폭별 `test_acc` = test 에피소드 50개 `acc` 평균 (< 1e-6) | 11 에폭 모두 50개, 최대 차 0.0 (valid도 50개, 최대 차 0.0) | 충족 |
| 3. A vs B 콘솔 에폭별 정확도 | 33개 전부 같음, `==>` 줄 같음. tqdm 진행 표시·`Final Result` 시각을 뺀 콘솔 전체도 같음 | 충족 |
| 4. A vs T01 4단계 Amazon_clothing 로그 | 33개 전부 같음, `==>` 줄 같음 | 충족 |

## 4. 이상 징후
1. 지시서의 변경 목록에 없는 `argument.py` `config2string` 제외 목록에 `"out_dir"`를 추가했다. 추가하지 않으면 `--out_dir` 미지정 시 `None == False`가 거짓이라 `[Config]`·`# Current Settings` 줄에 `out_dir_None`이 붙는다. 이 변경은 출력 문자열에만 관여한다.
2. `classes` 값이 `3.0` 같은 float로 저장된다(Amazon_clothing 클래스 목록 원소가 float). 값은 변환 없이 `.item()`만 적용했다.
3. `wall_time_sec`은 `train()` 시작~파일 작성 직전 구간이다. 데이터 로딩·구조 특징 생성(`embedder.__init__`)은 포함하지 않는다.
4. `episode_records`·`epoch_records`는 `--out_dir` 유무와 관계없이 메모리에 쌓이고, 파일 작성만 `out_dir`로 갈린다.
5. 두 run 모두 CLAUDE.md §4 개정 형식(UUID 지정)으로 실행했다. 지시서 명령의 `CUDA_VISIBLE_DEVICES=0`과 같은 물리 GPU다(T01 표 2).
