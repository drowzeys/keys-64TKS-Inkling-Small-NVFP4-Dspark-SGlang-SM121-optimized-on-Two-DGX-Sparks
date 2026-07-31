# Constitution

1. Correctness before speed: no config ships without the byte-exact lossless gate.
2. Every tok/s number carries its task class and concurrency. No bare numbers.
3. Patches are byte-exact file overlays from a validated image, pinned to an upstream digest.
4. Site specifics are env knobs; champion flags are immutable in scripts.
5. Failures get root causes, not workarounds — and get recorded in BUGS-AND-FIXES.md.
