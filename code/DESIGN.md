# Design: interfaces and file formats

This is the contract between the modules of `optsplit`. `README.md` explains how to run things.

## Paths

- `OPTSPLIT_ROOT` (env var, default `./workdir`): everything that is not code.
  - `$ROOT/data/<dataset>/`: prepared datasets (see below).
  - `$ROOT/runs/<suite>/<name>/`: one directory per run.
  - `$ROOT/jobs/<suite>/`: Slurm job lists and logs.
- A run is identified by `<suite>/<name>`, its path relative to `$ROOT/runs`.

## Config

A run config is `configs/defaults.yaml`, deep-merged with a base config (`configs/base/*.yaml`), then
with overrides (`key.sub=value`, values parsed as YAML). `optsplit.config.load_config(path, overrides)`
returns a `Config` (a dict with attribute access). The resolved config is saved as
`<run>/config.yaml`; `python -m optsplit.train <run>/config.yaml` runs it (and resumes it if
`latest.pt` exists).

Per-optimizer hyperparameters live under `optim.<name>`, and `optimizer` selects one, so a swap only
changes `optimizer`. `shadow` lists optimizers whose state is accumulated during this run with learning
rate 0 (the state update of SGD momentum, AdamW and Muon does not depend on the learning rate, and with
lr 0 the parameters are untouched), so a fork can start any of them warm.

`ghost` (optional, default none) lists update rules run *open-loop* beside the run (`optim.Ghost`): each is a copy of
the model, from the same initialization, updated by that rule (own state, hyperparameters, schedule) on the run's
gradients, so the copy never feeds back. A ghost of the run's own rule reproduces the run. Checkpoints go to
`<name>__ghost_<rule>/` (its `config.yaml` has `optimizer: <rule>` and `ghost_of`), at the run's `log.ckpt_steps`. Not
with forks. `optim.muon.orthogonalize: false` (Nesterov momentum rescaled to the Frobenius norm of an orthogonal update) and
`sv_power: p` (U S^p V^T from an exact SVD of the momentum, same rescaling; p=0 exact orthogonalization, p=1 the former;
for p<0 the singular values are first divided by the largest and those below 1e-6 raised to 1e-6, so S^p stays finite)
are Muon ablations. `two_track_steps: K` (with `sv_power` -0.3 or -0.5) replaces the SVD by 4K-1 matrix products
(`sv_power_two_track` in `optim/muon.py`): exact enough for K = 12, while fewer steps leave out the smallest singular
values. `lc_steps: K` (5 or 6, with `sv_power: -0.3`) does the same with 3K products (`sv_power_lc`): a weighted
sum of Newton-Schulz iterates, S^p above 3e-3 (K = 5) or 1e-3 (K = 6) of the Schatten-8 norm, less below. `lc_steps: 5m`
(LC5m, 15 products) keeps every output singular value between Muon's and S^p (`scripts/lc/lc5m.py`). `lc_steps: gram`
(Gram LC, `sv_power_gram`, `scripts/lc/gram_lc.py`) does the same from a polynomial of the Gram matrix X X^T, applied to
X once: 17 small-side and 2 rectangular products (0.84 of Muon's FLOPs for the 14M GPT), float32, S^p down to about
1e-4 of the Schatten-8 norm instead of 3e-3.
`log.two_track_check` lists steps at which a run logs the error of that update against the exact one
(`two_track` records: per Muon matrix, cosine and relative error for K = 5, 8, 12 and LC5, LC6).

Smoothed Muon (`optimizer: smuon`, hyperparameters `optim.smuon`: those of `optim.muon` plus `smooth: a`) is Muon whose
step on each matrix is the running average of Muon's updates O_t: U_t = (1 - a) U_{t-1} + a O_t, W_t = W_{t-1} - lr
U_t / (1 - (1 - a)^t) (a = 1 is Muon; `optim.muon.smooth` sets the same option, default off). U and t are kept in the
optimizer state (`smooth_buffer`, `smooth_step`), so they are saved and restored with it. A fork into smuon from a
parent that kept no smuon shadow starts from the parent's Muon state, and the average restarts (t = 0).

`train.loss` is `ce` (default) or `mse`: the squared error between the outputs z (the logits, no softmax) and the
one-hot target, summed over the classes and averaged over tokens (Cohen et al. 2021). It is the loss everywhere:
training, the probe's `train_loss` and `val_loss`, `delta_loss`, every Hessian and gradient measurement
(`model.loss`), the loss on the targets in the analysis. With `mse` the predicted distribution is
p = max(z, 0) normalized to sum 1 (uniform if no entry is positive), since the minimizer of the loss is the
conditional probability vector itself (`optsplit.loss`).

Step grids (`log.eval_steps`, `log.ckpt_steps`, `measure.steps`) are `{every: N, geometric: K, extra: [...]}`:
the union of `0, N, 2N, ...`, `K` log-spaced steps in `[1, max_steps]`, the `extra` steps, and
`max_steps`. `optsplit.config.step_grid(spec, max_steps) -> sorted list[int]`. `log.fork_steps` is a plain
list of steps.

Step `s` always refers to the model after `s` updates (step 0 is the initialization).

## Data (`optsplit.data`)

`build_data(cfg, device) -> Data` with:

- `get_batch(step, batch_size=None, seed=None) -> (x, y)`: training batch for `step`. Depends only on
  `(seed or cfg.train.data_seed, step)`, so forks and resumed runs see the same batch order.
- `fixed_batch(split, n, seed) -> (x, y)`: a fixed set of `n` examples from `split` (`train`/`val`),
  identical across runs. Used for the Hessian batch and the probes.
- `random_batches(n_batches, batch_size, seed) -> list[(x, y)]`: training batches from a separate RNG
  stream (used for batch sharpness and gradient covariance; do not perturb the training order).
- `kind`: `"lm"` or `"image"`; `vocab_size` (lm) or `num_classes` (image).

LM: `x`, `y` are int64 `(B, T)` token ids; `y` is `x` shifted by one. Image: `x` float `(B, 3, 32, 32)`
(normalized), `y` int64 `(B,)`.

Prepared data: LM datasets have `train.bin`, `val.bin` (uint16 token ids) and `meta.json`
(`vocab_size`, plus `itos` for characters or `tokenizer.json` next to it). CIFAR-10 has
`train_x.npy` (uint8 N×32×32×3), `train_y.npy`, `test_x.npy`, `test_y.npy`.

## Models (`optsplit.models`)

`build_model(cfg) -> nn.Module` (GPT or SimpleCNN). Both have:

- `model(x) -> logits` (GPT: `(B, T, V)` for every position; CNN: `(B, C)`).
- `model.loss(x, y) -> scalar` mean loss (`train.loss`; `model.loss_kind`, set by `build_model`).
- GPT also has `model.forward_embeds(tok_emb) -> logits`, where `tok_emb = model.embed(x)` is the
  `(B, T, d)` token-embedding output (position embeddings are added inside). Input-derivative metrics
  differentiate with respect to `tok_emb`; for the CNN the input is the image itself.
- `param_blocks(model) -> dict[block -> list[param name]]` partitions all parameters for the
  anatomy of the top Hessian eigenvector. GPT: `emb` (wte, wpe), `L<i>.qk`, `L<i>.ov`, `L<i>.mlp`, `norms`,
  `head`. CNN: one block per layer.
- `muon_params(model) -> (hidden, other)`: names of parameters Muon updates (hidden matrices, conv
  filters) and those its auxiliary AdamW updates (embeddings, head, gains, biases).

Attention uses `F.scaled_dot_product_attention`. Code that needs second or third derivatives runs under
`optsplit.models.double_backward_ok()`, which forces the math SDPA backend. All measurements run in
float32 without autocast, on the uncompiled module.

## Optimizers (`optsplit.optim`)

`build_optimizer(name, model, cfg) -> torch.optim.Optimizer`; each param group keeps its base learning
rate in `group["base_lr"]` and `set_lr(opt, mult)` sets `lr = base_lr * mult`. `lr_mult(step, cfg)` is the
schedule (linear warmup, then cosine to `min_lr_frac` or constant). `Shadow(model, cfg)` builds the
optimizers in `cfg.shadow` with lr 0; `shadow.step()` after the real step updates their state.
SGD and AdamW are `torch.optim.SGD/AdamW`; Muon is `SingleDeviceMuonWithAuxAdam` (vendored from
KellerJordan/Muon). AdamW state is `state[p]["exp_avg_sq"]`, `state[p]["step"]`.

## Run directory

- `config.yaml`: resolved config.
- `metrics.jsonl`: one JSON object per line, with `kind` and `step`. On resume, records may repeat;
  readers keep the last one per `(kind, step)`.
  - `train`: `loss`, `lr`, `grad_norm`, `time`.
  - `eval`: `train_loss` (fixed train probe, used for loss matching), `val_loss`, `val_acc`.
  - `measure`: whatever `optsplit.measure.measure_step` returns (below), plus `lr`, `delta_loss`.
- `preds/<step:07d>.pt`: `{"logp": float16 (N, V)}`, log-probabilities on the fixed validation probe
  (LM: `probe.n_seqs` sequences × `probe.positions_per_seq` fixed positions; image: `probe.n_seqs`
  test images). `preds/targets.pt`: `{"y": int64 (N,)}`. With `train.loss: mse` the file is `{"z": float16 (N, V)}`,
  the raw outputs; `analysis.io.Run.logp` derives the log-probabilities from them and `Run.outputs` returns z.
- `ckpt/<step:07d>.pt`: `{"model": state_dict}` (float32 weights) at `log.ckpt_steps`. Step 0 is always
  saved (needed for distance from initialization). With `log.ckpt_optimizer`, `opt/<step:07d>.pt`:
  `{"optimizer": state_dict}` of the run's optimizer at the same steps.
- `fork/<step:07d>.pt`: `{"model", "optimizer": {name: state_dict}}` with the run's own optimizer and
  every shadow optimizer, at `log.fork_steps`.
- `latest.pt`: resume state. `done`: written when the run finished.

A fork run has `fork.parent = "<suite>/<name>"` and `fork.step = s`. It loads the parent's
`fork/<s>.pt`, takes `optimizer[cfg.optimizer]` from it (the parent's own state if the optimizer is the
same, otherwise the shadow state), and continues from step `s` to `max_steps` with the same batch order
(unless `train.data_seed` is overridden). Its own files only cover steps `>= s`.

## Measurements (`optsplit.measure`)

`measure_step(model, optimizer, cfg, data, step) -> dict` (flat-ish, JSON-serializable), called at
`measure.steps` before the update of that step:

- `lambda_max`, `eigs` (top `measure.n_eigs`): top Hessian eigenvalues of the loss on the fixed Hessian
  batch (`data.fixed_batch("train", measure.hessian_batch, seed=1)`), by Lanczos.
- `precond_lambda_max` (AdamW only): top eigenvalue of `P^{-1/2} H P^{-1/2}`,
  `P = (1 - beta1^t) diag(sqrt(v_hat) + eps)` (Cohen et al. 2022).
- `batch_sharpness`: `E_B[g_B^T H_B g_B] / E_B[||g_B||^2]` over `measure.batch_sharpness_batches` random
  mini-batches of the training batch size (Andreyev & Beneventano 2024, Def. 1).
- `anatomy`, `anatomy_precond`: for the top eigenvector `v`, `{"dir": {block: ||v_b||^2}, "self": {block:
  v_b^T H_bb v_b}, "cross": lambda - sum_b self_b}`. (`v_b^T (Hv)_b` is just `lambda ||v_b||^2`, so the
  sharpness is split into each block's own curvature along `v` and the cross-block remainder.)
- `gradcov` (if `measure.gradcov`): Jastrzebski et al. (2020) quantities from `measure.gradcov_L`
  mini-batch gradients: `lambda_K` (top eigenvalue of K), `lambda_K_min` (smallest non-zero),
  `trace_K`, `g_over_g5` (||g|| / ||g projected on top-5 eigenvectors of K||).

Thresholds (computed in analysis from `lr` and the config): SGD with momentum `mu`: `(2 + 2 mu)/lr`;
AdamW preconditioned: `(2 + 2 beta1)/((1 - beta1) lr)`.

## Offline metrics (`optsplit.metrics`)

`python -m optsplit.metrics.run <run> [--steps ...]` loads `ckpt/<step>.pt` files and writes
`<run>/function_metrics/<step:07d>.json` with input-derivative metrics, weight norms and robustness.

## Analysis (`optsplit.analysis`)

Reads run directories only (never imports training code except to build models for LMC). Writes figures
and tables to `$ROOT/analysis/<name>/`.
