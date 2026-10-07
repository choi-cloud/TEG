# 정리 기록 — 2026-10-07 (사용자 결정)

- **삭제:** `results/T10/` 전체(파일 590개, 그중 `emb_best.npy` 74개). 디스크 사용량 `du -sh` 971M(apparent size 945M).
- **삭제 근거:** CLAUDE.md §7(H1) 규칙. TN2 T10의 부분 결과(2026-10-07 00:07–00:13 KST, 디스크 부족으로 중단)로, T10r로 대체되어 사용하지 않는다.
  - 대체 실험 T10r의 보고서가 커밋·push됐고, 태그 `t10r`(`6a16717`)이 원격에 있다.
  - 사용자 지시로 `emb_best.npy`만이 아니라 디렉터리 전체를 지웠다.
- **다른 `results/`:** 변경하지 않았다.
- **`df -h /` (2026-10-07 20:03 KST):**
  - 정리 전: `/dev/mapper/ubuntu--vg-ubuntu--lv   98G   81G   12G  88% /`
  - 정리 후: `/dev/mapper/ubuntu--vg-ubuntu--lv   98G   80G   13G  87% /`
