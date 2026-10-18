# GPN 업스트림 출처 (T24)

- 저장소: https://github.com/kaize0409/GPN_Graph-Few-shot (Ding et al., CIKM 2020)
- 커밋: `40ef635aa2a81f175e56c5588e1d50688f6aa85e` (`git clone --depth 1`, 2026-10-18)
- 바이트 그대로 복사한 파일

| 파일 | sha256 |
|---|---|
| `models.py` | `7105c20625213080bcc1d9e308e4a98ad57c6d2f596bb875bc2aa3f7811b37cc` |
| `layers.py` | `1580692202121069bcc9edc3e434a56fb66eab54bc7374c05c76d08bec1c4ee2` |

- `train_gpn.py`, `utils.py`는 복사하지 않고 참조만 한다(`tools/run_t24.py`에 학습 스텝을 수식 그대로 옮김).
- `models.py`는 `from layers import GraphConvolution`을 쓰므로, 불러올 때 `tools/gpn/`을 `sys.path`에 넣는다.
