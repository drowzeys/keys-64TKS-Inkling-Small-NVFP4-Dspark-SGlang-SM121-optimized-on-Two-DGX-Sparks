# Agent instructions

You are installing Inkling-Small-NVFP4 + DSpark on two DGX Sparks (GB10/sm_121a) via SGLang.

1. Read `specs/001-oneshot-install/spec.md` (what & success criteria), then execute
   `specs/001-oneshot-install/tasks.md` **in order, respecting every GATE**.
2. Do not "fix" boot failures from first principles before checking
   `docs/BUGS-AND-FIXES.md` — all 12 known walls are listed with exact fixes.
3. Do not substitute MoE/attention backends: marlin + triton are the only correct pair on sm_121.
4. The lossless gate (T4) is mandatory. Fluent-but-different output at temp 0 = broken numerics.
5. Site-specific values (IPs, NIC, RDMA device, model path) are env knobs on the launch script —
   never edit the champion flags themselves.
