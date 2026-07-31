# Plan

1. Preflight both nodes (RDMA visible, link up, docker+CUDA13) — T0
2. Stage 161 GB of weights once, share via NFS to both nodes — T1
3. Bake `local/sglang-inkling:gb10` per node: digest-pinned upstream + 7 patch groups — T2
4. Launch rank1 → rank0 with champion defaults — T3
5. Gate on losslessness before anything else — T4
6. Bench and compare to published numbers — T5
7. Point clients at :30000/v1 — T6

Rationale for every non-obvious choice lives in docs/BUGS-AND-FIXES.md; the launcher encodes
the champion config as defaults so the happy path is a two-command boot.
