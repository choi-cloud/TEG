# T22 — 대조(InfoNCE)를 전체 노드 풀(`--pres_pool all`)로 재실행

2026-10-15 · 브랜치 `pool_all` · H0 `84c3522` · 지시서 `instructions/T22_contrast_pool_all.md` · 요약표 `reports/T22_summary.md`

## 1. 변경 파일과 위치

| 파일 | 내용 |
|---|---|
| `tools/run_t22.py` (신규) | `tools/run_t10r.py` 복사. 설정 7개: `base`(`--pres_lambda 0 --pres_pool nb`), `infonce_nb_l{10,30,100}`(`--pres_pool nb --pres_loss infonce`), `infonce_all_l{10,30,100}`(`--pres_pool all --pres_loss infonce`). 모든 run에 `--pres_pool` 명시. 공통 `--dump_logits --fixed_eval`, seed 0–2만 `--dump_emb`(이때 `emb_best.npy`도 완료 조건). 마감 시각 로직 제거, 디스크 여유 1.5 GB 점검 유지 |
| `tools/analyze_t22.py` (신규) | `tools/analyze_t10r.py` 복사. 설정·표·판정을 T22 §3–§5로 교체. 풀별 λ 선택, 게이트 Gb–Gd·Gf, 표 T1–T5, K1–K6, P112–P115. Gf 미충족이거나 `--no_verdict`이면 판정·사전 등록 대조를 쓰지 않음 |
| `reports/T22_pool_all.md`, `reports/T22_summary.md` | 보고서·요약표 |

- `model.py`, `utils.py`, `argument.py` 변경 없음. `--pres_pool all`은 기존 경로(`model.py:81–85`). 새 난수 없음(풀 표본 7000 + seed, 증강 8000 + seed 기존 생성기).
- Gf 기준(사용자 확정): 셀마다 \|T22 nb seed 평균 − T10r seed 평균\| ≤ 2.0 %p(고정 test, T10r 선택 λ 10 / 30 / 30 / 10 / 100 / 30, `reports/T10r_summary.md` 표 1에서 읽음). seed 단위 차는 T5 기록용.
- K3 정의: 고정 에피소드 짝지은 차 C_all − C_nb의 평균의 절댓값 ≤ 1.0 %p인 셀 수. 기록값 "분명히 음수" = 평균 < 0이고 \|평균\| > 2·SE.

## 2. 실행 명령과 출력

| 단계 | 명령 | 시간 | 출력 |
|---|---|---|---|
| 학습 | `python tools/run_t22.py` (GPU 0, 1, 2, 4, 5, 6, UUID, 동시 6) | 03:53:14–04:38:57 (45.5분) | 420/420 성공, 재시도 0 |
| 분석 | `python tools/analyze_t22.py` | 04:39:04–04:40:52 (1.8분) | `reports/T22_summary.md`, `results/T22/gate_checks.txt` |

- 디스크: `results/T22/` 1.6G(그중 seed 0–2 `emb_best.npy` 126개 1.1G). 리포 디스크 여유 8.8G(실행 후).

### 게이트

| 게이트 | 결과 |
|---|---|
| Ha | `t21` 있음. T10r `infonce_l{10,30,100}` seed 0–9 `run.json`·`fixed_eval_logits.npz` 각 60/60 |
| Gb | (셀, seed) 60개 × 설정 7개: `eval_logits.npz`(전 에폭)·`fixed_eval_logits.npz` 에피소드 키 불일치 0 |
| Gc | `ckpt_recheck` `epoch == ckpt_epoch`, 50/50: 420/420 |
| Gd | `config.pres_pool`·`pres_lambda`가 설정 이름과 일치, λ > 0 run의 `pres_loss` = infonce, 에폭별 `pres_loss_mean` 유한: 실패 0 |
| Ge | 부기 커밋에 기록 |
| Gf | 셀별 \|T22 nb − T10r\| = 0.006 / 0.002 / 0.014 / 0.000 / 0.000 / 0.002 %p (clothing 1, 5 / dblp 1, 5 / electronics 1, 5). 충족 |

### 선택 λ (nb / all)

| 셀 | clothing 1 | clothing 5 | dblp 1 | dblp 5 | electronics 1 | electronics 5 |
|---|---|---|---|---|---|---|
| λ | 10 / 10 | 30 / 10 | 30 / 10 | 10 / 10 | 100 / 100 | 30 / 30 |

### 판정과 사전 등록 (값만; 표는 `reports/T22_summary.md`)

| 판정 | 값 |
|---|---|
| K1 (C_all − TEG 분명히 양수) | 6/6 |
| K2 (C_all − D 분명히 양수; 평균 > 0) | 5/6; 5/6 |
| K3 (\|C_all − C_nb\| ≤ 1.0 %p; 분명히 음수) | 5/6; 4/6 |
| K4 (F(C_all) − F(base) 분명히 양수) | 6/6 |
| K5 (기록, C_all − F(base) 분명히 양수) | 4/6 |
| K6 (기록, 선택 λ 같은 셀) | 4/6 |

| 예측 | 조건 | 관측 | 일치 |
|---|---|---|---|
| P112 | K1 = 6/6 | 6/6 | 일치 |
| P113 | K2(분명) ≥ 4/6 | 5/6 | 일치 |
| P114 | K3 ≥ 5/6 | 5/6 | 일치 |
| P115 | K4 = 6/6 | 6/6 | 일치 |

- K2에서 C_all − D 평균 ≤ 0인 셀: Amazon_clothing 5-shot(−0.35 ± 0.12).
- K3에서 \|C_all − C_nb\| > 1.0 %p인 셀: Amazon_electronics 1-shot(−1.62 ± 0.14). 분명히 음수인 셀: dblp 1, dblp 5, electronics 1, electronics 5.

## 3. 완료 조건 대비

| 조건 | 결과 | 충족 |
|---|---|---|
| H0 커밋(브랜치 `pool_all`) | `84c3522`(지시서만, 이상 징후 1) | 예 |
| 420 run, 모든 run에 `--pres_pool` 명시 | 420/420, Gd 통과 | 예 |
| seed 0–2 `--dump_emb` | `emb_best.npy` 126개 | 예 |
| 표 T1–T5 | `reports/T22_summary.md` | 예 |
| 게이트 Ha–Gf | 전부 충족(Ge는 부기 커밋) | 예 |
| K1–K6, P112–P115 대조 | 위 표 | 예 |
| 실행 시간·디스크 기록 | 2절 | 예 |
| `results/T22/` 유지 | 지우지 않음 | 예 |

## 4. 이상 징후

1. 착수 시 `git status`에 `CLAUDE.md`(modified)가 없었다. `CLAUDE.md`의 T22 관련 수정은 H0 전에 `e9af265`("CLAUDE.md: 2026-10-15 edits before T22", `instructions/T22_CLAUDE_md_edits.md`에 따른 수정)로 이미 커밋돼 있었다. H0 `84c3522`에는 지시서만 들어갔고, 사용자 지시로 `84c3522`을 Ge 기준으로 쓴다.
2. T22 `infonce_nb`와 T10r `infonce`(T10r 선택 λ)의 고정 test 에피소드별 정확도 벡터가 seed 60개 중 49개에서 완전히 같고 11개에서 다르다. seed 단위 차 최대 \|0.08\| %p.
3. `base`는 지시서대로 `--pres_pool nb`를 명시했다. λ = 0이라 풀을 만들지 않는다(`model.py:76`).
4. 선택 λ가 격자 끝(10 또는 100)인 셀: nb 3개(clothing 1 = 10, dblp 5 = 10, electronics 1 = 100), all 5개(clothing 1, clothing 5, dblp 1, dblp 5 = 10, electronics 1 = 100). 표 T1의 ‡.
5. F(cfg) 온도 격자 끝 최적값 run 0개.
