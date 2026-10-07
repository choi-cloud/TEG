# T15 보고 — 0단계: 코드 사실 보고 (실행 없음, 2026-10-08, 브랜치 `erasure_diagnosis`, manual mode)

## 1. 변경 파일과 커밋
- 착수 커밋 **H0** = `fa547c5` (`fa547c521a9b20d027c92738229b60d47b649aa8`), 메시지 `docs: CLAUDE.md 개정(2026-10-08) + T15 지시서`.
  - 변경 파일: `CLAUDE.md`(사용자 제공 개정본으로 교체), `instructions/T15_code_facts.md`(추가).
  - 부모 `66909a1`(`blind_spot_contrast`).
- 보고서 커밋: 이 파일만 추가(`report: T15 code facts`). 태그 `t15`. push 결과는 별도 부기 커밋에 적는다.
- 조사 대상: H0의 코드. 아래의 `파일:줄`은 모두 H0 기준이다.

## 2. 사용한 조사 명령 목록
- python 실행 0회(import 확인 포함). 코드·설정 파일 변경 0건.

| 구분 | 명령 |
|---|---|
| 착수 점검 | `git status --porcelain`, `git branch --show-current`, `git rev-parse --short HEAD`, `git merge-base --is-ancestor cecc750 HEAD`, `git branch --list erasure_diagnosis`, `git ls-remote --heads origin erasure_diagnosis` |
| H0 | `git switch -c erasure_diagnosis`, `git add CLAUDE.md instructions/T15_code_facts.md`, `git commit`, `git log --oneline -1`, `git rev-parse HEAD` |
| 첨부 파일 위치 확인(착수 전) | `ls -la instructions/`, `find / -xdev -name ... -newer reports/T14_bsc_E1.md`(파일 이름 검색), `ls -t`, `date` |
| 읽기 | Read 도구: `instructions/T15_code_facts.md`, `layers/EGNN.py`, `embedder.py`, `utils.py`, `model.py`(40–769행). `git diff CLAUDE.md`, `cat -n configuration.yaml`, `sed -n`(`tools/analyze_t06b.py`, `tools/analyze_t14.py`, `tools/fusion_baseline.py`), `wc -l memory.py` |
| 검색 | `grep -n`(`model.py`, `argument.py`, `layers/*.py`, `memory.py`, `bsc_sampler.py`, `main.py`, `utils.py`, `embedder.py`, `tools/analyze_t06b.py`, `tools/analyze_t12.py`, `tools/analyze_t13.py`, `tools/analyze_t14.py`, `reports`, `docs`), 설치된 라이브러리 소스 `torch_geometric/nn/conv/gcn_conv.py`, `torch_geometric/nn/dense/linear.py`, `torch/optim/optimizer.py`, `torch/optim/adam.py` |
| 저장소 상태 | `git branch`, `git tag`, `df -h /`, `du -sh results/*`, `find results/<실험> -name emb_best.npy`(개수·용량) |

## 3. 완료 조건 대비
| 조건 | 결과 | 충족 |
|---|---|---|
| A–J 작성, 각 사실에 `파일:줄` | 4절 A–J. 라이브러리 내부·데이터 파일 의존 사실은 "확인 불가" 또는 "정적 확인으로 확정 불가"로 표기 | 충족 |
| 코드 파일 변경 0 | `git diff --stat fa547c5 <보고서 커밋>` = `reports/T15_code_facts.md`만(부기에 기록) | 부기 |
| python 실행 0회 | 2절 | 충족 |
| 금지어 없음 | CLAUDE.md §6 금지어 6개를 `grep -nE`로 검색, 0건 | 충족 |
| 태그 `t15`, push, `git ls-remote` | 부기 커밋에 기록 | 부기 |

## 4. 조사 결과 A–J

### A. 손실 항 전체
학습 1 step = `train_epoch("train", ...)`의 에피소드 1회(`model.py:449–578`).

| 항 | 코드 위치 | 수식(코드 그대로) | 입력 텐서 | 참여 노드 | 사용 레이블 | 기본 가중치 |
|---|---|---|---|---|---|---|
| L_N (`loss_l1_train`, "Network Loss") | `model.py:562`, 출력 `model.py:529` → `episode_forward` `model.py:125–146` | `loss_fn(output, label_list)`, `loss_fn = torch.nn.NLLLoss()`(`model.py:423`), `output = F.log_softmax(-dists_output, dim=1)`(`model.py:143`), `dists_output = euclidean_dist(embeds_qry, embeds_proto)`(`model.py:142`) | EGNN 출력 좌표(`self.egnn(...)`의 두 번째 반환, `model.py:128`). EGNN 입력은 GCN 출력의 support·query 행(`model.py:471–478`)과 구조 특징(`model.py:481–484`) | support(n_way·k_shot), query(n_way·qry). 에피소드 그래프는 query↔support 양방향 간선만(`model.py:497–505`) | `label_list`(`model.py:556`): query의 에피소드 내 클래스 위치. 학습 모드에서는 base 클래스 | γ = 0.5(`argument.py:30`), 곱셈 `model.py:565` |
| L_G (`loss_l2_train`, "Graph Embedder Loss") | `model.py:563`, 출력 `model.py:488–493` | `loss_fn(output_gcn, label_list)`, `output_gcn = F.log_softmax(-dists_gcn, dim=1)`, `dists_gcn = euclidean_dist(gcn_qry, proto_embeds_gcn)`, `proto_embeds_gcn = gcn_spt.mean(1)` | GCN 출력(`embeddings`, `model.py:471`)의 support·query 행. LayerNorm·EGNN 없음 | support, query | 위와 같은 `label_list` | 1 − γ = 0.5(`model.py:565`) |
| 합 | `model.py:565` | `loss_train = self.args.gamma * loss_l1_train + (1 - self.args.gamma) * loss_l2_train` | — | — | — | — |
| 보조 손실(공통) | `model.py:567–571` | `loss_train = loss_train + self.args.pres_lambda * pres` (`pres_lambda > 0`일 때만) | `pres = self.pres_loss(embeddings)`(`model.py:569`), 분기 `model.py:195–201` | 풀 노드(아래) | 아래 | λ 기본 0.0(`argument.py:44`) → 기본 꺼짐 |
| `kl`(보존, `kl_h2`는 `--pres_teacher_hops 2`) | `model.py:315–329` | `kl = logp_t.exp() * (logp_t.masked_fill(eye, 0.0) - logp_s.masked_fill(eye, 0.0))`; `kl.sum(1).mean()`. `s_s = (z @ z.T / pres_tau)`, `z = F.normalize(self.egnn.LayerNorm(embeddings[idx]), dim=1)` | 학생: 같은 step의 GCN 출력(`model.py:471`) → `egnn.LayerNorm`. 교사: 고정 텐서 `pres_teacher` = 행 정규화 Â^hops X(`model.py:21–33`, `model.py:57–58`) | 풀에서 m = 1,024개(`model.py:318–319`, 생성기 7000 + seed). 풀 `nb` = 레이블이 base 클래스 집합에 없는 노드(`model.py:59–62`) | 손실 값 계산에 레이블 없음. 풀 구성에 전체 노드 레이블의 base 소속 여부 사용(`model.py:59–62`) | λ(지시서 지정). τ 기본 0.1(`argument.py:45`) |
| `mse` | `model.py:274–280` | `((z @ z.T - t @ t.T)[off] ** 2).mean()` | kl과 같은 z, t | 풀에서 m개(`model.py:276`, `_pres_sample` `model.py:203–205`) | 풀 구성만 | λ |
| `infonce`(대조, T14 BSC 경로 포함) | `model.py:297–313` | `0.5 * (one_side(h1, h2) + one_side(h2, h1))`, `one_side`: `denom = torch.logsumexp(torch.cat([between, refl], dim=1), dim=1)`, `(denom - between.diagonal()).mean()`, 온도 0.5 | `h = F.normalize(self.pres_head(self.egnn.LayerNorm(self._view_embedding()[idx])), dim=1)`. `_view_embedding`은 증강 입력의 **추가 GCN forward**(`model.py:282–295`) | 배치 `idx`: `_pres_batch`(`model.py:251–272`). `uniform`은 `_pres_sample`(7000 + seed), `emb_rwr`·`bsc_rwr`은 RWR(14000 + seed) | 손실 값 계산에 레이블 없음. `bsc_rwr` 배치 구성에 base 레이블 프로토타입 사용(`model.py:230`) | λ. 심판 마스크 `--referee_k`(기본 0, `argument.py:55`) |
| weight decay | `model.py:50` | 손실 항으로 더해지지 않음. `optim.Adam(..., weight_decay=5e-4)`. torch Adam이 `grad = grad.add(param, alpha=weight_decay)`(`torch/optim/adam.py:366–367`) | 모든 파라미터 | — | — | 5e-4 |

- **γ:** 인자 `--gamma`, 기본 0.5(`argument.py:30`). 곱해지는 위치는 `model.py:565` 한 곳이다.
- **L_N:** EGNN 출력(최종 좌표)을 쓴다. EGNN 입력 좌표는 GCN 출력이다(`model.py:128`, `layers/EGNN.py:70`).
- **L_G:** GCN 출력만 쓴다(`model.py:488–493`). EGNN과 LayerNorm을 거치지 않는다.
- **valid 모드의 손실:** valid 모드에서도 L_N·L_G를 계산한다(`model.py:558–565`). backward는 `mode == "train"`에서만 한다(`model.py:573–576`). `loss_epoch = 0`(`model.py:443`)은 이후 사용 위치가 없다(`grep` 1건).
- **에폭 0:** backward·step 없이 `zero_grad`만 한다(`model.py:577–578`).

### B. GCN 파라미터로 가는 gradient 경로
- **파라미터:** 기본 `--gcn_layers 1`이면 `GCN`(`layers/GCN.py:7–16`)이고 `GCNConv(in_dim, 64, cached=True, normalize=True)`(`layers/GCN.py:10`)다.
  - `conv1.lin.weight`: (64, F). `Linear(..., bias=False, weight_initializer='glorot')`(`torch_geometric/nn/conv/gcn_conv.py:210–211`).
  - `conv1.bias`: (64,)(`gcn_conv.py:214`).
  - F는 데이터셋 특징 차원이다. 정적 확인으로 확정 불가(데이터 파일 의존). T06b 보고서는 in = 9034/7202/8669를 기록했지만(`reports/T06b_coverage_gap_controls.md:30`), 데이터셋별 대응은 확인 불가다.
  - 출력 차원 64는 `configuration.yaml:1` `gcn_out`이다.
- **forward 범위:** 전체 그래프다. `self.conv(self.features, self.edges)`(`model.py:471`)는 전체 노드 특징과 전체 간선을 쓴다.
- **학습 1 step의 GCN 호출 수**
  - 기본·kl·mse: 1회(`model.py:471`).
  - infonce: 3회(`model.py:471` 1회 + `_view_embedding` 2회, `model.py:300–301` → `model.py:292`).
  - 그 밖에 에폭마다 `bsc_epoch_setup`이 1회(`model.py:229`, `no_grad`) 호출한다. infonce λ > 0이고 에폭 ≥ 1일 때다.
- **경로**
  - **L_N:** GCN(`model.py:471`) → 행 선택 `embeddings[id_support]`, `[id_query]`(`model.py:475–478`) → `egnn.LayerNorm`(`layers/EGNN.py:70`) → `gcl_0`, `gcl_1`(`layers/EGNN.py:71–76`) → 프로토타입·유클리드 거리(`model.py:133–142`) → `log_softmax`(`model.py:143`) → NLL(`model.py:562`).
    - 각 EGCL에서 좌표는 `coord2dist`(`layers/EGNN.py:41–46`)로 `coord_diff`, `sqr_dist`가 되고, `msg_mlp`(입력: 구조 특징 양 끝 + `sqr_dist`, `layers/EGNN.py:15–19`) → `trans_mlp` → `w`(`layers/EGNN.py:23`) → `x + diff_mean(coord_diff * w)`(`layers/EGNN.py:26–28`)로 간다.
    - 구조 특징은 `posi_model`(`layers/EGNN.py:32–39`)이 `msg`로 갱신하고, 다음 층 `msg`에 들어간다.
  - **L_G:** GCN(`model.py:471`) → 행 선택(`model.py:475–478`, `488–489`) → `mean(1)`(`model.py:491`) → 유클리드 거리 → `log_softmax` → NLL(`model.py:563`).
  - **kl / mse:** GCN(`model.py:471`, 같은 텐서) → `embeddings[idx]` → `egnn.LayerNorm` → `F.normalize` → 손실(`model.py:320–329` / `277–280`).
  - **infonce:** 증강 입력 GCN(`model.py:292`) → `[idx]` → `egnn.LayerNorm` → `pres_head`(`model.py:71`, `77`) → `F.normalize` → 손실(`model.py:300–313`).
- **gradient 차단 연산 (위치 — 대상)**
  - `model.py:74` — `pres_head` 초기화 값 대입(`no_grad`).
  - `model.py:177` — `dump_embedding`의 GCN forward.
  - `model.py:228` — `bsc_epoch_setup`의 z·프로토타입·그래프.
  - `model.py:268` — `pair_stats` 진단.
  - `model.py:343` — `eval_episode` 반환값 `.detach()`. 호출부 `model.py:366`은 `no_grad`.
  - `model.py:571` — `pres.detach()`(기록용 float).
  - `model.py:582–583` — 정확도용 `output_softmax`, `label_list`의 `.cpu().detach()`. backward 뒤에 실행된다.
  - `model.py:590` — logit 덤프.
  - `model.py:690` — valid·test `train_epoch` 전체.
  - `model.py:767–768` — 체크포인트 `state_dict` 복사.
  - `memory.py:46`, `memory.py:72` — 관계 메모리(기본 off).
  - `requires_grad=False` 사용 위치는 없다(`grep` 0건).
  - `structural_features`는 상수 텐서로 만들어진다(`embedder.py:80`).

| 항 | GCN에 gradient 도달(정적 판단) | 근거 줄 | 동적 확인 필요 여부 |
|---|---|---|---|
| L_N | 도달(EGNN 좌표 입력 경유, 차단 연산 없음) | `model.py:471–478`, `529`, `562`, `layers/EGNN.py:70–76`, `26–28` | 크기·0 여부는 동적 확인 필요 |
| L_G | 도달(직접) | `model.py:471`, `488–493`, `563` | 크기는 동적 확인 필요 |
| kl·mse | 도달(같은 forward 텐서, LayerNorm 경유) | `model.py:569`, `320`, `277` | 크기는 동적 확인 필요 |
| infonce | 도달(추가 GCN forward 경유, `fork_rng`는 gradient를 막지 않음) | `model.py:291–292`, `300–301` | 크기는 동적 확인 필요 |
| weight decay | 손실 경로가 아닌 optimizer 내부 항으로 GCN 파라미터 갱신에 들어감 | `model.py:50`, `torch/optim/adam.py:366–367` | 아니오 |

### C. anchor와 노드 참여
- **정체:** 가상 anchor 노드 16개다(`--anchor_size` 기본 16, `argument.py:32`). 각 anchor의 인덱스는 `len(features) + i`(`embedder.py:48`)이고, 원래 노드와 간선으로 이은 그래프(`edges_hub`)를 만든다(`embedder.py:45–59`).
- **선택:** anchor i는 전역 `random.sample(list(range(len(self.features))), num_sample_node)`(`embedder.py:53`)로 고른 노드와 연결된다. num_sample_node = N/2^(i+1)이고 최소 1이다(`embedder.py:49–52`).
- **출처 노드 집합:** 전체 노드(0…N−1)다. 클래스 제한은 없다(`embedder.py:53`).
- **역할**
  - 구조 특징 = 각 anchor로부터의 최단 경로 거리 d에 대해 1/(d+1)이고, 도달하지 못하면 0이다(`embedder.py:62–80`). 노드당 16차원이다.
  - EGNN의 `str_feature` 입력으로 쓰인다(`model.py:481–484`, `layers/EGNN.py:51`, `55`). `msg_mlp` 입력 차원은 16 + 16 + 1이다(`layers/EGNN.py:10`, `model.py:48`).
  - 학습 파라미터가 아니고 손실 식에 직접 들어가지 않는다. anchor 노드 자체는 에피소드·손실에 참여하지 않는다.
  - 간선 목록에 나오지 않는 노드가 nx 그래프에 포함되는지는 정적 확인으로 확정 불가다(`embedder.py:62–64`, `add_edges_from`만 사용).
- **학습 step에서 임베딩이 손실에 직접 들어가는 노드**
  - L_N·L_G: support·query뿐이고, 학습 모드의 클래스는 `class_list_train`(base)이다(`model.py:455`, `463`). `class_list_train`은 train 클래스에서 valid를 뺀 것이다(`utils.py:67–69`).
  - 보조 손실(λ > 0): 풀 노드다. `nb` 풀은 레이블이 base 집합에 없는 노드(valid·test 클래스 등, `model.py:59–62`)이고, `all` 풀은 전체 노드다.
  - 입력 쪽: support·query의 GCN 출력은 1-hop 이웃(클래스 무관)의 특징을 집계한다(`layers/GCN.py:15`, 전체 간선 `model.py:471`). gradient는 support·query 행에서 `conv1` 파라미터로 간다.
  - `.mat` 파일에 없는 노드는 레이블 0으로 남는다(`utils.py:36–38`). 이런 노드가 있는지, 레이블 0이 실제 클래스인지는 정적 확인으로 확정 불가다(데이터 의존). 그 노드가 `nb` 풀에 드는지도 마찬가지다.

### D. 최적화와 정칙화
- **optimizer:** `optim.Adam`(`model.py:50`). lr과 weight decay는 모든 그룹에서 같다.
  - 그룹 0 `self.conv.parameters()`, 그룹 1 `self.egnn.parameters()`. lr = `--lr`(기본 0.001, `argument.py:12`), weight_decay = 5e-4(코드 상수, 플래그 없음).
  - infonce에서는 그룹 2 `self.pres_head.parameters()`를 추가한다(`model.py:78`). lr·weight_decay는 지정하지 않아 생성자 기본값(0.001, 5e-4)이 들어간다(`torch/optim/optimizer.py:1028–1034`).
- **dropout:** GCN 입력 특징에 `F.dropout(x, p=self.dropout, training=self.training)`(`layers/GCN.py:14`)이고 p = `--dropout` 0.5(`argument.py:22`)다. EGNN에는 dropout이 없다(`layers/EGNN.py` 전체).
- **LayerNorm:** `nn.LayerNorm(in_dim=64)`(`layers/EGNN.py:66`). EGNN 입력 좌표에 1회 적용하고(`layers/EGNN.py:70`), 보조 손실(`model.py:277`, `300–301`, `320`)과 BSC 진단(`model.py:229`)에서도 같은 모듈을 쓴다.
- **없는 것:** 그 밖의 정규화(BatchNorm 등), gradient clipping, scheduler는 없다(`grep` 0건).
- **step 수:** 1 에폭 = `--episodes` 50 step(`argument.py:26`)이다. 에폭은 0…`--epochs`(10)의 11개(`model.py:684`, `argument.py:13`)이고, 에폭 0은 업데이트가 없다(`model.py:574–578`). 업데이트 step은 500이다.
- **zero_grad·step 위치:** `zero_grad`는 학습 에피소드 시작(`model.py:452`)과 에폭 0의 끝(`model.py:578`)에 있다. `backward`·`step`은 `model.py:575–576`(에폭 ≠ 0)이다.
- **train/eval 전환**
  - 학습: 에폭 ≥ 1이면 `train()`, 에폭 0이면 `eval()`(`model.py:425–431`).
  - valid·test: `eval()`(`model.py:433–435`).
  - `bsc_epoch_setup`은 `eval()` 후 `train()`으로 되돌린다(`model.py:226–227`, `247–248`).
  - `dump_embedding`은 `conv.eval()`만 한다(`model.py:176`).
  - `run_fixed_eval`은 `eval()`(`model.py:362–363`), `mem_sanity_check`도 `eval()`(`model.py:413–414`)이다.

### E. 레이블 사용 지점
| 위치 | 읽는 레이블 | 용도 분류 | 내용 |
|---|---|---|---|
| `utils.py:30–38`, `44–53` | 전체(train·test `.mat`) | 학습 계산 | `labels`, `id_by_class`(클래스별 노드 목록) 생성 |
| `utils.py:67–69` | train 클래스 목록 | 학습 계산 | valid 클래스 분할(전역 `random`) |
| `model.py:455`, `463`, `556`, `562–565` | base | 학습 계산 | 학습 에피소드 노드 선택, query 레이블 → L_N·L_G |
| `model.py:458`, `463`, `556`, `562–565`, `746–749`, `756–769` | valid | 모델 선택 | valid 에피소드 정확도 → `best_acc_valid`, `test_acc_at_best_valid` 시점, `emb_best.npy` 덤프 시점, 체크포인트 시점. valid 손실도 계산하지만 사용하지 않음 |
| `model.py:461`, `463`, `556`, `582–585` | test | 지표 | test 정확도·F1 |
| `model.py:59–62` | 전체 노드 레이블(base 소속 여부) | 학습 계산 | 보조 손실 풀(`nb`) 구성(λ > 0) |
| `model.py:213–214`, `230` | base(+ valid 클래스 목록으로 제외) | 학습 계산(`bsc_rwr`) / 지표(`uniform`·`emb_rwr`) | base 프로토타입 → 풀 그룹 배정 |
| `model.py:269`(`bsc_sampler.pair_stats`) | 배치 노드(풀 = valid·test) | 지표 | `fn_rate` |
| `model.py:367–372` | test, valid | 지표 | 고정 에피소드 `query_y` |
| `model.py:370`, `377–385` | test | 지표 | 체크포인트 재평가 |
| `model.py:193` | 전체 | 지표 | `labels.npy` 저장(분석용) |
| `model.py:93`, `104–105` | 에피소드 노드 | 지표 | 관계 메모리(`--mem`, 기본 off) |

### F. 무작위성 소비 지점
| 위치 | 생성기 | 내용 |
|---|---|---|
| `main.py`의 `seed_everything(set_seed)` → `utils.py:259–267` | torch CPU·CUDA, numpy 전역, `random` 전역 | seed 설정 |
| `utils.py:67` | 전역 `random` | valid 클래스 분할 |
| `embedder.py:53` | 전역 `random` | anchor 연결 노드 16회 |
| `model.py:45`/`47`, `48` | torch 전역(CPU) | GCN(glorot, `torch_geometric/nn/dense/linear.py:30–31`), EGNN(`nn.Linear` 기본 초기화, `layers/EGNN.py:9` xavier) 파라미터 초기화 |
| `model.py:455`, `458`, `461`; `utils.py:277` | 전역 `random` | 에피소드 클래스·노드 추출(학습·valid·test 모두) |
| `layers/GCN.py:14` | torch 전역(입력 텐서 장치의 생성기) | 학습 모드 dropout. 장치별 생성기 구분은 정적 확인으로 확정 불가 |
| `model.py:66`, `204`, `318` | `torch.Generator` 7000 + seed | 보존·uniform 풀 표본 |
| `model.py:69`, `75–76`, `285–286` | `torch.Generator` 8000 + seed | `pres_head` 초기화, 증강 마스크 |
| `model.py:70–71` | torch 전역, `fork_rng` 안 | `pres_head` 생성 시 기본 초기화(곧 8000 생성기 값으로 덮어씀) |
| `model.py:291–292` | torch 전역, `fork_rng` 안 | 증강 view GCN forward(학습 모드 dropout) |
| `model.py:215`, `258`, `261` | `np.random.default_rng` 14000 + seed | BSC 에폭 0 균일 추출, RWR |
| `model.py:225` | `fork_rng` | BSC 에폭 설정(eval 모드) |
| `model.py:368` | `random.Random` 9000 / 9100 + seed | 고정 test / valid 에피소드 |

- 그 밖의 `np.random` 전역 호출은 `model.py`, `utils.py`, `embedder.py`, `layers/*.py`, `memory.py`, `bsc_sampler.py`, `main.py`에 없다(`grep`, `utils.py:266`의 seed 설정 제외).

### G. 임베딩 덤프와 측정 도구
- **`emb_best.npy`**(`model.py:174–193`)
  - 텐서: `self.conv(self.features, self.edges)`, 즉 GCN 출력이다. LayerNorm 전, EGNN 전, 전체 노드 [N, 64] float32(`model.py:178`, `180`).
  - 모드: `conv.eval()`, `no_grad`(`model.py:176–177`). `fork_rng`는 쓰지 않는다.
  - 시점: 각 에폭의 valid·test 뒤 `acc_valid == best_acc_valid`일 때 덮어쓴다(`model.py:761–762`, `--dump_emb`, `argument.py:41`). 남는 파일은 이 조건을 만족한 마지막 에폭의 것이고, 에폭 번호는 `run.json` `emb_epoch`(`model.py:181`)에 남는다.
  - 처음 저장할 때 `split.json`, `labels.npy`도 저장한다(`model.py:182–193`).
- **에폭별 덤프:** 현재 플래그로는 불가다(`argument.py`에 해당 인자 없음). 덤프 위치는 `model.py:761–762` 한 곳이다.
- **초기화 직후 덤프(첫 optimizer step 전)**
  - 별도 플래그는 없다.
  - 에폭 0은 업데이트가 없고(`model.py:574–578`), 에폭 0에서 `best_acc_valid`가 0에서 갱신되므로 에폭 0 끝에 `dump_embedding(0)`이 호출된다. 이 파일은 이후 조건을 만족하는 에폭에서 같은 파일명으로 덮어쓴다.
  - 에폭 0의 valid 정확도가 0이면 조건이 거짓이 된다. 이 경우를 포함해 실제 저장 여부는 정적 확인으로 확정 불가다.
- **kNN 순도(T14)**
  - 위치: `tools/analyze_t14.py:56` `knn_purity(Z, nodes, y, k=10, block=4096)`.
  - 입력: `Z` numpy [N, d], `nodes` 그룹 노드 인덱스, `y` 그 노드의 레이블. 행 L2 정규화 후 그룹 안 코사인 top-k의 같은 레이블 비율 평균이다.
  - 레이블: 대상 그룹의 레이블(T14에서는 test).
  - import: 함수로 정의돼 있다. 다만 모듈 최상위에서 `fusion_baseline`, `analyze_t13`(→ `analyze_t06b`: `os.chdir(ROOT)`, `from utils import load_data`, `tools/analyze_t06b.py:21–24`)을 import한다. `main()`은 `__main__` 가드 안(`tools/analyze_t14.py:286`)에 있다.
- **그룹별 코사인 프로토타입 probe(T12)**
  - 함수: `tools/analyze_t06b.py:90` `episodes(rng, classes, by_class, k)`, `:103` `probe_acc`, `:116` `fewshot_acc(V, eps, k)`. 5-way, query 5, 500 에피소드, support 평균 프로토타입, 코사인이다.
  - 입력: `V` numpy [N, d], `random.Random(1000 + seed)` 인스턴스, 그룹 클래스 목록과 `by_class`.
  - 레이블: 그룹(base·valid·test) 노드 레이블(`by_class` 구성).
  - 그룹 구성은 `tools/analyze_t12.py:124–131`의 `main()` 안 인라인 코드다(함수로 import 불가). 같은 구성은 `tools/analyze_t13.py:75` `groups_of(run_dir)`에 함수로 있다.
  - import: `analyze_t06b` 모듈은 import 시 `os.chdir(ROOT)` 등이 실행된다(`tools/analyze_t06b.py:21–24`). `main()`은 `__main__` 가드 안(`:463`)에 있다.
- **유효 차원·BW(T13)**
  - 함수: `tools/analyze_t13.py:44` `erank(M)`, `:52` `erank_node(Z)`, `:56` `group_metrics(Z, nodes, y)`(반환: `erank_node`, `erank_cls`, `BW`, `C`, `n`, `spec`), `:75` `groups_of(run_dir)`(`split.json`, `labels.npy`).
  - 입력: `Z` numpy [N, d], 그룹 노드, 레이블.
  - 레이블: 그룹 노드 레이블(`erank_cls`, `BW`).
  - import: 함수로 정의돼 있다. 모듈 최상위에서 `analyze_t06b`, `analyze_t12`를 import한다(`tools/analyze_t13.py:20–21`). `main()`은 `__main__` 가드 안(`:249`)에 있다.

### H. 진단 스위치 가능성 (계획만)
- **3번의 전제:** EGNN을 거치지 않는 항으로 L_G가 이미 있다(`model.py:488–493`). `--gamma 0`이면 학습 손실은 L_G만 남는다(`model.py:565`). 다만 정확도·모델 선택·고정 평가는 EGNN 출력(`model.py:529`, `582`, `342`)을 쓴다. 그래서 γ 설정만으로는 "평가도 GCN 프로토타입"이 되지 않는다.

| # | 스위치 | 현재 플래그로 가능한가(인자) | 불가하면 바꿀 파일·함수 | 기본(꺼짐) 경로를 원본과 같게 유지 | 전역 `random` 소비에 영향 | 비고 |
|---|---|---|---|---|---|---|
| 1 | L_N만(L_G 끔) | 가능: `--gamma 1.0`(`argument.py:30`, `model.py:565`) | — | 기본값 변경 없음 | 없음(손실 산술만) | 0·L_G 항은 계산은 됨. L_G 쪽 gradient가 정확히 0인지는 동적 확인 필요 |
| 2 | L_G만(L_N 끔) | 학습 손실은 가능: `--gamma 0.0` | — | 기본값 변경 없음 | 없음 | EGNN 파라미터는 손실 gradient 0이어도 Adam weight decay로 갱신됨(`model.py:50`). 평가는 EGNN 출력 사용 |
| 3 | EGNN 끈 TEG(학습·평가 모두 GCN 프로토타입) | 불가 | `argument.py`(새 플래그), `model.py` `train_epoch`(정확도 출력 선택 `model.py:529`/`582`, 손실 `565`), `eval_episode`(`model.py:331–343`) | 플래그 꺼짐이면 같은 코드 경로 유지 가능(정적 판단) | EGNN 생성(`model.py:48`)을 유지하면 초기화 난수 소비 동일. `random` 호출 순서 변경 없음 | 최소 변경 단위: 학습은 `--gamma 0`, 평가 출력을 `output_gcn` 쪽 softmax로 바꾸는 분기 2곳(`train_epoch`, `eval_episode`) |
| 4 | EGNN → GCN stop-gradient | 불가 | `model.py` `train_epoch`: `episode_forward`에 넘기는 `embeds_epi`(`model.py:529`)를 플래그로 `detach()` | 가능(플래그 꺼짐 시 같은 텐서) | 없음 | L_G·보조 손실 경로는 그대로 GCN에 도달 |
| 5 | base 레이블 섞기(손실용 레이블만) | 불가 | `model.py` `train_epoch` `label_list`(`model.py:556`)를 학습 모드에서만 치환 | 가능(플래그 꺼짐 시 같은 계산) | 섞기에 난수가 필요. 전용 생성기(§8 등록부 15000 이후 미배정)를 쓰면 전역 `random` 영향 없음(정적 판단) | "섞기"의 단위(에피소드 내 query 레이블 순열 / 노드-클래스 대응 전역 순열)는 지시서에서 정해야 함 |
| 6 | 지도 손실 0(정칙화·최적화만) | 불가(γ 하나로 두 항을 동시에 0으로 만들 수 없음) | `argument.py` + `model.py:565`(지도 손실 계수 0 분기) | 가능 | 없음(forward·에피소드 추출 유지 시) | Adam weight decay만으로 파라미터가 갱신됨(`torch/optim/adam.py:366–367`) |
| 7 | 임베딩 차원 64 → 128, 256 | 불가(CLI 인자 없음). `configuration.yaml:1–2` `gcn_out`, `egnn_in` 값 변경 필요 | `configuration.yaml` 또는 `argument.py`·`main.py`(설정 덮어쓰기 인자). EGNN `in_dim` = `egnn_in`과 GCN 출력이 일치해야 함(`model.py:47–48`) | 기본 64 유지 시 같음 | 전역 `random` 영향 없음. torch 전역 초기화 소비량이 바뀌어 이후 torch 난수열(dropout)이 기본과 달라짐(정적 판단) | `pres_head` 입력 `conf["gcn_out"]`(`model.py:71`), 출력 64 고정 |
| 8 | LayerNorm 끔 | 불가 | `layers/EGNN.py:70`(`EGNN.forward`) 분기. 보조 손실·BSC의 `self.egnn.LayerNorm` 사용처(`model.py:229`, `277`, `300–301`, `320`)는 별도 결정 필요 | 가능(모듈 생성 유지) | 없음(LayerNorm 초기화는 상수) | — |
| 9 | weight decay 0 | 불가(상수 5e-4, `model.py:50`) | `argument.py` 새 인자, `model.py:50` | 기본값 5e-4 유지 시 같음 | 없음 | `pres_head` 그룹도 기본값을 따름 |
| 10 | 에폭별·초기화 시점 임베딩 덤프 | 불가(G절) | `model.py` `train()`(`model.py:684` 루프 앞 / 에폭 끝), 별도 파일명. `dump_embedding`(`model.py:174–193`)은 `conv.eval()` 후 모드를 되돌리지 않음 | 가능(플래그 꺼짐 시 호출 없음) | eval 모드 forward는 dropout 난수를 쓰지 않음. CLAUDE.md §3은 진단용 forward를 `fork_rng` 안에서 하도록 규정 | 초기화 시점 = 에폭 0 학습 전 또는 에폭 0 끝(에폭 0에 업데이트 없음) |

### I. 저장소 상태 (H0 직후, 2026-10-08)
- **브랜치:** `erasure_diagnosis`(현재), `HT_01`, `blind_spot_contrast`, `dual_view_trust`, `main`, `novel_like_class`, `preserve_rehearse`, `t11_wip`.
- **태그:** `t01`, `t02`, `t03a`, `t03b`, `t04`, `t05`, `t06`, `t06b`, `t07`, `t08`, `t08b`, `t09`, `t10r`, `t12`, `t13`, `t14`.
- **HEAD:** H0 = `fa547c5`.
- **`df -h /`:** `/dev/mapper/ubuntu--vg-ubuntu--lv 98G 82G 12G 89% /`.
- **`du -sh results/*`:** _diag.txt 0, T01 152K, T02 180K, T03a 708K, T03b 1.1M, T04 15M, T05 184M, T06 166M, T06b 335M, T07 1.3G, T08 14M, T08b 8.0K, T09 2.9G, T10r 398M, T12_gate_checks.txt 4.0K, T13_gate_checks.txt 4.0K, T14 2.1G.
- **`emb_best.npy`가 남아 있는 실험(지우지 않음)**

  | 실험 | 파일 수 | 용량 |
  |---|---|---|
  | T06 | 18 | 159M |
  | T06b | 36 | 317M |
  | T07 | 126 | 1.1G |
  | T09 | 288 | 2.5G |
  | T14 | 210 | 1.9G |

### J. 정적 확인으로 확정하지 못한 항목과 동적 확인 방법 (실행하지 않음)
| 항목 | 동적 확인 방법 |
|---|---|
| 데이터셋별 GCN 입력 차원 F | 데이터셋마다 `load_data` 후 `features.shape[1]`, 또는 `conv1.lin.weight.shape` 출력 |
| 각 손실 항이 GCN 파라미터에 주는 gradient의 크기·0 여부 | 같은 step에서 항별로 `torch.autograd.grad(term, conv.parameters(), retain_graph=True)`의 norm 기록 |
| `--gamma 1.0`/`0.0`에서 꺼진 항의 gradient가 정확히 0인지 | 위와 같은 항별 gradient 비교 |
| `.mat`에 없는 노드(레이블 0)의 존재 여부와 `nb` 풀 포함 여부 | `labels`, `data_train["Index"]`, `data_test["Index"]` 비교 후 개수 출력 |
| nx 구조 특징 그래프에 간선 없는 노드가 포함되는지, 구조 특징이 0인 노드 수 | `structural_features`의 0 행 개수 출력 |
| 에폭 0 끝의 `dump_embedding(0)` 호출 여부 | 에폭 0 `acc_valid` 값과 `dump_embedding` 호출 기록 |
| dropout이 소비하는 생성기(CPU/CUDA)와 `fork_rng(devices=[device])`의 보존 범위 | 호출 전후 `torch.cuda.get_rng_state()`·`torch.get_rng_state()` 비교 |
| 차원 변경(H-7) 시 이후 난수열 변화 범위 | 기본·변경 설정에서 같은 seed의 에피소드·dropout 마스크 비교 |

## 5. 이상 징후
1. **착수 점검 1(`git status` 깨끗함):** 문자 그대로는 맞지 않았다. 작업 트리에 `M CLAUDE.md`, `?? instructions/T15_code_facts.md`(사용자 첨부본)가 있었다. 멈추고 확인을 요청했고, 사용자가 "첨부본 맞음, 진행"으로 답한 뒤 4–6단계를 진행했다. 점검 2(브랜치 `blind_spot_contrast`, HEAD `66909a1`)와 점검 3(`cecc750` 조상)은 맞았다.
2. **조사 명령의 범위:** 2절의 `find`, `sed -n`, `cat -n`, `wc -l`, `date`는 허용 목록(`git`, `ls`, `du`, `df`, `grep`/`rg`, 파일 읽기)에 이름으로 없다. 모두 파일 이름 검색·파일 읽기·개수 세기·시각 확인 용도다.
3. **태그 표기:** 개정 CLAUDE.md §1의 "태그" 줄은 `t07`, `t08b`, `t09`, `t10r`, `t12`, `t13`, `t14`이고, 저장소의 태그는 `t01`–`t14` 16개(I절)다.
4. **`dump_embedding`의 모드:** `dump_embedding`(`model.py:176`)은 `conv.eval()`만 하고 `egnn` 모드와 이후 모드를 되돌리지 않는다. 다음 `train_epoch` 시작에서 모드가 다시 설정된다(`model.py:425–435`).
5. **valid 손실:** valid 모드에서도 L_N·L_G를 계산하지만(`model.py:558–565`) 그 값은 사용되지 않는다.
