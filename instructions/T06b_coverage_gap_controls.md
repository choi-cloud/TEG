# T06b — coverage gap 보강 진단: 조건 맞춘 대조군, 축 상실 판정, TEG와 같은 에피소드 비교 (브랜치 `novel_like_class`)

정본 `CLAUDE.md`. 선행: T06(`t06`). 이 파일은 `instructions/T06b_coverage_gap_controls.md`에 둔다.
**auto mode: CLAUDE.md §2의 멈춤 지점 두 곳을 이 지시서 동안 해제한다.** §5 게이트에서 실패하면 멈춘다. CLAUDE.md §7은 유효.

## 0. 목표 (검증할 주장 세 가지)
T06의 세 빈틈을 메운다.
1. **검사 ①** — T06의 "학습이 base 구분은 얻고 novel 구분은 잃는다"가 **학습 효과**인지, 64차원 압축·hop 수 차이 때문인지 가른다(hop·차원을 맞춘 학습 없는 대조군).
2. **검사 ②** — novel 클래스를 가르는 방향이 base가 쓰지 않는 축인지(축 상실), base와 같은 축인지(같은 축 위의 압축)를 **학습 없는 공간**에서 hold-out base 기준선으로 판정한다.
3. **검사 ③** — 학습 없는 기준선과 TEG를 **TEG의 실제 test 에피소드** 위에서 같은 조건으로 비교한다.

학습 경로는 원본 그대로다. 예외는 새 플래그 `--gcn_layers 2`를 켠 run뿐이다(별도 모델 변형).

## 1. 코드 변경
### 1.1 `--dump_test_eps` (store_true, 기본 False)
- valid·test 에피소드를 만들 때 이미 뽑힌 `id_support`, `id_query`, `class_selected`를 **읽어서** 쌓는다. mode `test`만 저장하면 된다.
- `--out_dir`가 있으면 학습 종료 시 `out_dir/test_eps.npz` 저장: `epoch`(E), `ep_idx`(E), `support`(E × N·K, 전역 노드 ID, 클래스 순서대로 K개씩), `query`(E × N·M, 같은 규칙), `classes`(E × N). E = 에폭 수 × 50.
- 새 코드에서 난수를 호출하지 않는다. 계산 경로를 바꾸지 않는다.

### 1.2 `--gcn_layers` (int, 기본 1)
- 1: 원본 `layers/GCN.py`와 **완전히 같은 모듈**(파라미터 이름·모양 동일).
- 2: dropout → GCNConv(in, 64) → ReLU → dropout → GCNConv(64, 64). 두 층 모두 `cached=True, normalize=True`, dropout은 `args.dropout`. 다른 하이퍼파라미터는 TEG 기본값 그대로.
- `--dump_emb`의 덤프 대상은 마지막 GCNConv 출력(LayerNorm·EGNN 이전).

### 1.3 공통
- 새 플래그 2개를 `config2string` 제외 목록에 추가.

## 2. 실행 (36 run)
- 1층: 3 데이터셋 × 5-way {1, 5}-shot × seed {0, 1, 2} = 18 run
  `... --dump_emb --dump_test_eps --out_dir results/T06b/L1/<ds>_5w<k>s/seed<s>`
- 2층: 같은 18 run에 `--gcn_layers 2`
  `... --gcn_layers 2 --dump_emb --dump_test_eps --out_dir results/T06b/L2/<ds>_5w<k>s/seed<s>`
- `tools/run_t06b.py`(`run_t06.py` 복사), GPU 6장 UUID, GPU당 1 run, seed 우선 순서. `--mem` 사용 안 함.

## 3. 분석 (`tools/analyze_t06b.py` → `reports/T06b_summary.md`)
라벨은 분석에만 쓴다. 분석 난수: probe 에피소드 `random.Random(1000 + seed)`(T06과 같음), 무작위 투영 `np.random.default_rng(2000 + seed)`, base 분할 `random.Random(3000 + seed)`, 순열 `np.random.default_rng(4000 + seed)`.

### 3.1 표현 공간 9종
Â = `gcn_norm(edge_index, add_self_loops=True)`(T06과 같음). X = `load_data`의 특징.

| 이름 | 정의 | hop | 차원 | 학습 |
|---|---|---|---|---|
| `raw` | X | 0 | 원 차원 | 없음 |
| `diff1` | ÂX | 1 | 원 차원 | 없음 |
| `diff2` | Â²X | 2 | 원 차원 | 없음 |
| `rand1` | ÂX·R, R ~ N(0, 1/64), 원 차원 × 64 | 1 | 64 | 없음 |
| `rand2` | Â²X·R (같은 R) | 2 | 64 | 없음 |
| `pca1` | ÂX의 PCA 64성분(전체 노드로 적합, 중심화, randomized SVD) | 1 | 64 | 없음 |
| `pca2` | Â²X의 PCA 64성분 | 2 | 64 | 없음 |
| `emb1` | L1 run의 `emb_best.npy` | 1 | 64 | 있음 |
| `emb2` | L2 run의 `emb_best.npy` | 2 | 64 | 있음 |

학습 없는 공간은 shot과 무관하다. (데이터셋, seed)마다 한 번 계산해 재사용해도 된다.

### 3.2 검사 ① — 조건 맞춘 대조군
- probe: T06과 같음(5-way, K = shot, query 5, 에피소드 500개, support 평균 프로토타입, cosine 최근접). 클래스 그룹 G ∈ {base, valid, test}. 에피소드는 G마다 한 번 뽑아 모든 공간에 공유.
- `emb1`·`diff*`·`rand1`·`pca1`은 L1 run, `emb2`·`rand2`·`pca2`는 L2 run의 split으로 계산한다(같은 seed면 split이 같아야 한다 — §5 G3).
- **표 B1**: (데이터셋, shot) × 공간 9종 × G 3종 정확도, seed 평균 ± sd.
- **표 B3 — 학습 효과**: Δ = acc(emb_h, G) − acc(대조군, G), 대조군 ∈ {rand_h, pca_h}, h ∈ {1, 2}. 열: Δ_base, Δ_valid, Δ_test, (Δ_base − Δ_test). seed 평균 ± sd, 부호 일치 수.
- **표 B4 — hop 효과**: acc(diff2) − acc(diff1), acc(emb2) − acc(emb1), acc(pca2) − acc(pca1), G별.

### 3.3 검사 ② — 축 상실 판정
대상 공간: `diff1`, `diff2`, `pca1`, `pca2`, `emb1`, `emb2`.

절차(run 또는 (데이터셋, seed) 단위):
1. 클래스 프로토타입 μ_c = 클래스 전 노드 평균, L2 정규화.
2. base 클래스를 무작위로 반씩 A(적합), B(보류)로 나눈다. **20회** 반복.
3. A의 순서쌍(i ≠ j) 차이 μ_i − μ_j로 SVD → 누적 분산 ≥ 0.90을 만드는 최소 r개 주성분 → 투영 P_A.
4. 잔차 비율 ρ(d) = ‖d − P_A d‖² / ‖d‖². 비순서쌍(i < j)으로 평균:
   ρ̄_A(A 쌍, 점검용), ρ̄_B(B 쌍, **기준선**), ρ̄_T(test 쌍), ρ̄_V(valid 쌍, 참고 — 이 로더에서 valid는 원 train 클래스 풀에서 무작위로 뽑힌다).
5. G = ρ̄_T − ρ̄_B, 비율 = ρ̄_T / ρ̄_B.
6. **순열 검정**(분할마다): B ∪ test 클래스를 한 풀로 합치고, 그중 |test|개를 무작위로 "test"로 다시 지정해 G를 계산(1,000회). p = (순열 G ≥ 관측 G인 비율). 풀 안 모든 쌍의 잔차를 먼저 계산해 두고 인덱싱만 하면 된다.

**판정 통과 정의(사전 고정)** — (데이터셋, 공간)마다 seed 3 × 분할 20 = 60개 값으로:
- (i) G > 0인 비율 ≥ 0.95
- (ii) 평균 비율 ρ̄_T / ρ̄_B ≥ 1.20
- (iii) 분할별 p의 중앙값 < 0.05
- 세 조건 모두 만족 → "통과(novel이 base가 쓰지 않는 축으로 갈림)".

- **표 C**: (데이터셋, 공간)별 r 평균, ρ̄_A, ρ̄_B, ρ̄_T, ρ̄_V, G, 비율, G > 0 비율, p 중앙값, 판정. `emb`는 shot별로 따로 행을 둔다.
- **표 C-s (민감도)**: 같은 표를 누적 분산 0.80, 0.95, 그리고 r = 10 고정으로. 판정 열만.
- **표 C2 — principal angles(보조)**: 공간마다 P_A와 P_B 사이, P_A와 P_T 사이(P_T: test 순서쌍 차이로 같은 0.90 규칙) principal angle 코사인의 평균. 분할 20회 × seed 평균.

### 3.4 검사 ③ — TEG와 같은 에피소드
- run마다 `best_epoch_valid`의 test 에피소드 50개(`test_eps.npz`)를 그대로 쓴다.
- 평가: TEG 실제 에피소드 정확도(`episodes.jsonl`의 같은 epoch·ep_idx), 그리고 같은 support·query로 probe:
  - L1 run: `emb1`, `diff1`, `diff2`, `pca1`, `rand1`
  - L2 run: `emb2`, `diff2`, `pca2`, `rand2`
  - 각 probe는 cosine과 유클리드(제곱 거리) 두 가지.
- **표 D**: (층, 데이터셋, shot)별 방법 × 거리 정확도, seed 평균 ± sd, TEG 대비 짝지은 차(에피소드 50 × seed 3 = 150개 차의 평균 ± SE, 양수 비율).
- **표 D2 — 분해(L1)**: `diff1` probe → `emb1` probe → TEG 전체. 단계별 차이(유클리드, cosine 각각).

## 4. 사전 등록 (수정 금지, 2026-10-06 작성)
- **P42.** 표 B3: (Δ_base − Δ_test) > 0 이 `emb1 − rand1`, `emb1 − pca1` **둘 다에서** 6개 (데이터셋, shot) 중 4개 이상.
- **P43.** 표 B4: acc(diff2, test) > acc(diff1, test)가 6개 중 4개 이상.
- **P44.** 표 C: `diff1`과 `pca1`에서 "통과" 판정이 나오는 데이터셋이 각각 3개 중 1개 이하.
- **P45.** 표 D: TEG와 같은 에피소드에서 `diff2` cosine probe ≥ TEG(L1)가 6개 (데이터셋, shot) 중 3개 이상.

## 5. 게이트 (실패 시 새 코드 안에서만 수정, 최대 2회 → 그래도 실패면 멈추고 보고)
- **G1 — 기본 경로 불변**: `--gcn_layers 1`의 GCN 모듈 `state_dict` 키·모양이 원본(`t06`)과 같다. `git diff`의 추가 줄에 난수 호출이 없다. L1 run의 `test_acc_at_best_valid`를 T06 같은 seed 값과 나란히 기록(원본 비결정성 때문에 일치는 요구하지 않음).
- **G2 — 파일**: 36 run 모두 `emb_best.npy`, `test_eps.npz`, `split.json`, `labels.npy`, `run.json` 존재. `test_eps.npz`의 `best_epoch_valid` 에피소드 50개, 에피소드당 노드 수 = 5·(K + 5), `classes`가 `episodes.jsonl`의 같은 epoch·ep_idx와 일치.
- **G3 — 분석 점검**: 같은 seed의 L1·L2 split이 같다. 90% 규칙에서 ρ̄_A ≤ 0.12. 순열 G 분포의 평균의 절댓값이 관측 G 분포 sd보다 작다.
- **G4**: 표 B1, B3, B4, C, C-s, C2, D, D2 생성.

## 6. 보고 `reports/T06b_coverage_gap_controls.md` (사실만, 해석 금지)
1. 변경 파일·위치·줄 수, 커밋 해시, 태그 `t06b`
2. 게이트 G1–G4
3. 표 B1, B3, B4, C, C-s, C2, D, D2 본문
4. P42–P45 "예측 / 관측 / 일치·어긋남"
5. 이상 징후(자율 판단 포함)

## 7. 하지 말 것
CLAUDE.md §7 전부. 1층 경로 변경. 2층 run의 하이퍼파라미터 조정. 판정 기준·문턱 변경. 지정 외 공간·지표 추가. 결과 해석.
