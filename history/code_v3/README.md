# Frozen-decoder encoder adaptation snapshot v3

This is an independent snapshot copied from code_v2. It does not modify code_v1, code_v2, the source C2 checkpoint, or their active runs. It adds a bounded encoder adaptation test after constrained latent-search diagnostics found a large reachable reconstruction gap even under the original posterior-mean LayerNorm constraint.

## Two arms and the question they answer

Both arms import the exact same C2 full-training checkpoint: SHA256 `773d355ada1e7d1355d6e32047a3ce77a08ebfdba1455c22002513078fd590d6`, source step100000, source protocol `c41c23a9a018c40bd306386af871a4212926db1f9793a3b804c3ffcf79f09de8`. The original visual adapter, posterior and decoder weights are shared initially. The decoder is frozen throughout. Both use a new AdamW optimizer; this is an explicit new adaptation phase, not restoration of the previous optimizer or an implicit source-code migration.

| Arm | New mixer | Total network parameters | Trainable parameters | Frozen decoder |
|---|---|---:|---:|---:|
|baseline |none |1,304,358 |998,400 |305,958 |
|mixer2 |2 global self-attention blocks |2,883,878 |2,577,920 |305,958 |

The mixer follows `visual_adapter(features) + position_features` and precedes the original query posterior. Each block is pre-norm, width256,4 attention heads, FFN1024, GELU, dropout0. Attention output projection and the second FFN projection, including biases, are initialized to zero. There is no extra final normalization. The whole mixer is therefore exactly the identity initially; CPU tests verify bitwise-identical initial stochastic z, mean, logvar, KL and rendered RGB after resetting the same RNG. The additional1,579,520 parameters mean that any practical benefit cannot be attributed solely to global connectivity rather than added capacity.

The current posterior retains its original positional attention bias and LayerNorm mean readout. The learned posterior variance remains trainable. Both arms retain the stochastic reconstruction loss plus KL weight1e-5; neither changes to deterministic-mean training nor uses optimized held-out latents as targets.

## Budget and data

Prepared configurations, to run concurrently after each card's existing binding work finishes: baseline on physical GPU3 and mixer2 on physical GPU1, both H200 with no active neighboring workload. This is matched model-of-GPU and training-loop GPU-seconds, not a same-physical-card crossover. Keep per-card throughput/device mapping with the result:

- `../control/encoder_C2_base_s713.json`
- `../control/encoder_C2_mixer2_s713.json`

Each runs all9749 cached training clips, batch4,1024 sampled native128 RGB queries, seed713, LR1e-4, weight decay0.01, KL1e-5. Data, query and posterior-noise RNG streams are matched and separately restored. Each gets `--train-hours 0.25`:900 seconds of cumulative training-loop time on one H200, with at most one completed-update overshoot. Evaluation/checkpoint overhead is outside that budget and is separately reported. `--max-hours 0.4` is a wall-time safety bound, and the shared absolute deadline is08:45 Asia/Shanghai. An absolute deadline can truncate the experiment; inspect the recorded stop reason rather than assuming the target budget was reached.

Both perform an initial four-case evaluation, quick evaluations/checkpoints every5000 updates, and a final fixed32-clip evaluation. Metrics render at native128px in these configurations, so they must not be numerically mixed with historical64px quick-validation curves. Initial four-case MSE should reproduce the same C2 posterior-mean baseline. Any later generation-admission gate must use an additional standalone fixed-checkpoint evaluation on the original pilot32 cohort at64px; the native128 trial metrics do not replace that gate. Numeric checkpoints are retained with a100-checkpoint cap plus initial/best/latest/final aliases. The initial source100000 updates are recorded in the import identity; the new adaptation phase starts at update0. Report both incremental GPU-hours and updates, and distinguish the common source-training cost.

## Integrity and inference

`--source-warmstart` requires a fixed checkpoint SHA, original source-code directory and the exact original data/cache identities. The embedded source protocol and all bound source hashes are verified before importing weights. The geometry/config must match; only mixer parameters may be new. No source optimizer is restored. The new protocol binds source initialization, decoder fingerprint, encoder configuration, new source hashes, sampling policy and compute budget. Native v3 resume restores optimizer and all RNG streams; cross-snapshot loading goes through explicit source initialization only.

The decoder has `requires_grad=False`, remains in eval mode, and is fingerprinted before/after evaluations, every checkpoint and final status. Freezing its parameters still permits gradients through its latent input into the encoder. Standalone reconstruction evaluation, instance-binding evaluation and frozen-codec flow loading understand the new model configuration and verify the frozen decoder fingerprint.

For any later v3 flow run, the observed-prefix features pass through the same visual adapter, positional features and mixer using only the prefix tokens. Future features are used only by the labeled posterior training target / future-seen AE diagnostic. This snapshot does not modify or replace the existing v1 Wan flow run.

## Evidence boundaries and tests

All reconstruction figures have seen the true future. Better AE MSE or a smaller oracle-search gap does not establish prefix-only predictability, semantic object identity, or physical-law compliance. Compare held-out foreground MSE, purity/fragmentation, temporal JS and fixed-case GIFs before making an object-binding claim. A single seed and two sizes support a pilot decision, not a definitive architecture result.

The independent CPU contract tests use the real unmodified v1 Codec fixture: shared initial output, new-layer gradients, frozen-decoder integrity, exact checkpoint/optimizer roundtrip and tamper rejection. Run locally with:

```sh
python -m unittest discover -s code_v3 -p test_encoder_adaptation.py -v
```

On a remote layout, set `ENCODER_SOURCE_DIR=/absolute/path/to/original/code` first. Further CLI smoke evidence is stored in `../code_v3_smoke/`: explicit source import; two steps then resume to three matching uninterrupted three steps bitwise; standalone AE evaluation; frozen-codec CPU-toy flow; GT-mask binding using real RGB/masks with explicitly artificial VAE features; and a training-budget stop after exactly one completed update. These synthetic fixtures verify engineering only and are not model-quality results. No GPU work was launched by the snapshot builder.

`source_copy_manifest.json` records copied code_v2 bytes; `changes.patch` describes v3 runtime changes and the new independent test; `checksums.json` binds this final snapshot. Preserve the entire snapshot with its checkpoints.
