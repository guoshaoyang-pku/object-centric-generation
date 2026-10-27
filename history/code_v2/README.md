# Research code snapshot v2 — 2026-10-27

This is a separate immutable experiment snapshot. It does not replace remote `code/` or the sources bound into previous A/B/C2 checkpoints. Keep earlier runs with their original code. New experiments use a fresh output directory and this snapshot as a whole. No production source was modified for C3 integration.

## Variants

- A / xyt_t: existing xyt decoder with posterior xyz locality.
- B / xyt_xy: same decoder/initialization, removing only posterior time-distance bias.
- C / C2 / trajectory: persistent parent trajectory and affine geometry,109 two-dimensional local texture kernels by default;43+763=806 decoded fields.
- C3 / trajectory3d: same parent geometry with64 full3-D local(u,v,t) texture kernels;43+768=811 fields. Each texture kernel has3 center,6 covariance and3 RGB coefficients. Default trainable codec parameter count1,305,643, of which decoder307,243.

C3 inherits the frozen trajectory motion/local-coordinate/affine functions. Geometry parsing and weighting preserve the parent formulas. The 3-D texture follows the legacy packed-precision Gaussian formula and64-point DC subtraction; its spatial coordinates move with the object while time allows changing appearance. It is not a lifetime gate. C2→C3 changes the texture family, kernel count/layout and DC handling; it is not a perfectly isolated single-bit temporal ablation. Nor does time-varying texture establish object identity.

The mathematical static-material test exercises the broad-temporal-kernel limit of the texture formula; its near-zero temporal precision fixture lies outside the deliberately bounded learned covariance range. This is not evidence of exact static-texture expressibility for every fitted setting.

## Integration differences

`codec_research.py` imports the C3 class, adds explicit aliases/config admission, usesK64 by default forC3, rejects legacy-geometry warmstart for both trajectory variants, and includes the C3 source in checkpoint identity bindings. A/B/C2 weights and rendering remain unchanged under the same seed.

`evaluate_object_binding.py` dispatches C3 before its parent class and evaluates `contributions` + `texture3d` − `texture_dc`; each chunk is checked against the actual RGB renderer. GT future slice remains4:24 and native metrics128px.

`codec_flow_research.py` was copied from the final reviewed source SHA215fc731a784f386832870090365261173137457ce196b1d61f2a31cd2926bf0, then adds `trajectory_codec_3dtex.py` to its source bindings and the explicit variable-count construction described below. Its variant-aware policy is A/B=`legacy_grid`, C2/C3=`object_xy`; A/B have five time anchors at each xy, so xy alone would incorrectly make distinct time slots exchangeable. It enforces Diffusers0.35.2 and the actual Wan TI2V5B configuration before real-Wan use.

## Commands

Use existing validated FP32 VAE cache; no VAE or DiT is loaded by codec AE training. Example new run:

```sh
python code_v2/codec_research.py train --variant C3 --data DATA --cache CACHE --out NEW_RUN --batch 4 --points 1024 --steps 100000 --max-hours 0.5 --eval-count 32 --eval-size 128 --cohorts COHORTS --train-cohort pilot_train1024 --eval-cohort pilot32 --deadline-unix ABSOLUTE_DEADLINE
```

A new flow run accepts this C3 checkpoint through the same codec factory, preserves its source/data/cache identity, and freezes it. RGB denoising auxiliary defaults0. Real GPU smoke is still required; CPU toy backend is never a performance/quality result.

```sh
python code_v2/evaluate_object_binding.py --checkpoint CKPT --data DATA --cache CACHE --gt ANNOTATIONS --out NEW_BINDING_DIR --deadline-unix ABSOLUTE_DEADLINE
```

## Reproduction and validation

`source_copy_manifest.json` records the exact original source bytes. To reconstruct, copy its13 named files (11 runtime files and two CPU test files) from the recorded source revisions into `code_v2/`, then run `patch -p1 < code_v2/changes.patch` from its parent directory. `checksums.json` binds the final snapshot.

CPU checks passed:7 C3 mathematical tests; A/B/C2 same-seed state/render exact equivalence; C3 two-step then resume tostep3 identical to uninterruptedstep3; separate bound evaluation; C3 binding with verified real GT/RGB and explicit artificial-feature engineering fixtures; new C3 checkpoint admitted by the CPU toy flow for two train steps and fixed-seed previews. Native binding GIF20frames512x158. Evidence is under `../code_v2_smoke/`; fixture results carry no reconstruction/generation-quality claim. No GPU workload was launched by the snapshot integrator.

## Variable-count trajectory flow: C3 64 parents × 16 children

The existing C3 codec factory already supports this configuration without source changes. `make_wan_flow` now uses the historical 250 argument solely to construct count-independent Wan adapter modules, then wraps them with direct object tokens and the codec's explicit xy anchors. For N64 the transformer processes exactly64 tokens at every block; it never creates a fake250-token volume. The retained `legacy_grid=(5,5,10)` attribute is inactive in `object_xy` mode. Identity metadata distinguishes actual `processed_object_tokens` from `legacy_constructor_scaffold_tokens`. A/B retain their250 grid and reject other sizes before loading Wan. No codec/model/renderer source identity changed in this addition.

Flow attention/LoRA/projection parameters do not depend on object count. The frozen Wan backbone memory footprint therefore remains unchanged. Reducing250→64 tokens reduces the self-attention pair count to6.55%, and token-linear/cross-attention work to25.6% at fixed context; these are operation-count ratios, not measured end-to-end speedups. The existing full fingerprint and version gates still apply. An old250 flow checkpoint is intentionally not silently warmstarted into64; the current strict legacy architecture check rejects it. Start N64 from the pretrained Wan base plus fresh adapter/LoRA initialization, then resume its own bound checkpoint.

The new CPU tests use the actual WanObjectFlow, direct adapter and LoRA with a tiny independent backbone. They verify C3 N64/M16/K64/L128 posterior and RGB shapes, finite gradients and parameter counts; N64 and N250 direct paths with native volume/RoPE/Conv forward made fatal; the exact token count inside every transformer block; finite LoRA gradients; N64 checkpoint roundtrip and wrong-count rejection; and unchanged A/B admission. All19 snapshot tests pass. These are interface tests, not validation of pretrained-Wan quality or GPU throughput.

```sh
python -m unittest discover -s code_v2 -p 'test_*.py' -v
python code_v2/codec_research.py train --variant C3 --objects 64 --subkernels 16 --texture-kernels 64 --latent-dim 128 --data DATA --cache CACHE --out NEW_N64_RUN --batch 4 --points 1024 --steps 100000 --max-hours 0.5 --eval-count 32 --eval-size 128 --cohorts COHORTS --train-cohort pilot_train1024 --eval-cohort pilot32 --deadline-unix ABSOLUTE_DEADLINE
```

### Interpretation and small-goal protocol

This is a quality/binding/compute Pareto comparison, not matched representation capacity:

| C3 setting | Parents × children | Geometry + texture fields/parent | Fields/clip | Latent values | Codec parameters |
|---|---:|---:|---:|---:|---:|
|250 ×4, K64, L128 |1000 children |43 +768 |202,750 |32,000 |1,305,643 |
|64 ×16, K64, L128 |1024 children |115 +768 |56,512 |8192 |1,276,531 |

The nearly equal network parameter counts conceal a3.59× smaller decoded representation and3.91× fewer texture kernels/latent values. Child counts are similar but the parent sharing constraints differ. Lower H(parent|instance) alone is expected from the smaller alphabet: its maximum drops from log2(250)=7.97 to log2(64)=6 bits (the evaluator's extra unassigned bucket changes this slightly). Do not declare stronger binding from that decrease alone. Report both raw bits/effective parent count and normalized entropy as descriptive quantities; normalization does not remove the capacity confound.

First run a short same-four-training-clip overfit diagnostic with matched initialized seeds, query schedule and allocated H200 GPU-hours. Then compare the existing fixed32 held-out RGB clips plus the four verified held-out instance-mask clips at equal training GPU-hours. Keep the four mask-bearing training cases separate from held-out. Report native128 foreground/background RGB MSE, H(instance|parent) purity, H(parent|instance) fragmentation, visible adjacent-frame and reappearance-gap JS drift, background/unassigned mass, and same-case GT/AE/GT-mask/parent GIFs. Record training, evaluation and total allocated GPU-hours separately and measure examples/second and peak memory. Use posterior-mean future-seen AE labels; forecast quality requires a separate prefix-only flow experiment and is not inferred from this test.

An exploratory retention rule is: keep N64 as a useful Pareto point if foreground MSE is within10% of the N250 control at equal GPU-hours, H(instance|parent) does not worsen by more than0.1 bits, and either measured computation falls or temporal JS improves without coverage collapse. Treat these as pilot engineering thresholds, not statistical significance. Reject the semantic-binding claim if only fragmentation decreases, if parents merge distinct instances, or if lower JS comes with vanishing/diffuse reconstructions. Four held-out annotated clips support case diagnosis only; a general claim needs more independent clips and seeds. No result here establishes physical-law compliance.

### Current renderer limitation: no ordered occlusion layers

A/B/C2/C3 currently compute RGB as a normalized density mixture of parent colors. Parent index is a slot identity/anchor convention; it is not a front-to-back layer index, and there is no alpha/transmittance visibility computation. Two overlapping parents therefore blend colors instead of one naturally occluding the other. Reducing the number of parents can expose this limitation more strongly because each parent must cover more varied shape, appearance and occlusion states. This experiment does not implement the user's proposed ordered-latent renderer. Adding an untested alpha family tonight would confound the current capacity/binding diagnosis, so it is deliberately deferred.

Shared parent motion restricts child geometry, but a parent may still cover different semantic instances at different times, several parents may track fragments of one instance, and time-dependent texture can change appearance without preserving identity. Persistent parameters alone therefore establish neither persistent object identity nor physical dynamics. Correct collisions, contacts, occlusion and conservation require separate evidence; all current AE binding measurements use the true future and cannot establish predictive physical compliance.

A separate frozen-decoder latent-refinement probe can expose an amortization/latent-search gap by optimizing held-out true-future z. It is an oracle reconstruction diagnostic, not forecasting. Better refined MSE shows that this fixed decoder has a better reachable latent for that clip; it does not show that the posterior can predict it from the prefix, that a larger encoder will recover it, or that the learned prior can generate it. Failure of one finite optimization run is not a decoder capacity bound. Report latent displacement, actual completed steps, training/evaluation GPU-hours and the last evaluated step; do not substitute optimized-latent binding results for the trained posterior results.
