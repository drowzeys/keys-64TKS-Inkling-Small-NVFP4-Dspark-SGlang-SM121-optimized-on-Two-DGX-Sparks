# Spec: Lossless Inkling-Small + DSpark serving on 2× DGX Spark

## Goal
An OpenAI-compatible endpoint serving thinkingmachines/Inkling-Small-NVFP4 with the RadixArk
DSpark speculator, TP=2 across two GB10 nodes, that is (a) byte-exact lossless vs non-speculative
decoding at temp 0, and (b) ≥60 tok/s single-stream on raw continuation workloads.

## Success criteria
- SC1: server logs `Initialized DSpark draft runner ... gamma=7` (spec config not silently dropped)
- SC2: T4 lossless probe matches byte-exact
- SC3: raw continuation ≥55 tok/s, accept_len ≥6.5 (256 tok, temp 0)
- SC4: chat C1 medians within ±15% of README table
- SC5: survives C8 concurrent load without scheduler death (no wedge/EngineDead)

## Non-goals
- fa4 attention, TRT-LLM/cutlass FP4 MoE (arch-impossible / numerically unsafe on sm_121)
- Contexts >64K with full DSpark speedup (draft is 64K-adapted; serving more is fine, accept fades)
- vLLM path (Inkling MTP works there but caps at k=1 on 0.26-line builds; Lamport op needs MNNVL)
