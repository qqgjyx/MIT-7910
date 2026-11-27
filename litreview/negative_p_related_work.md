# Negative p (spectral powers): related work and where LC5m stands

Searched 2026-11-27. Every source was opened unless marked "not read".

## Our method and numbers

- **Update:** U S^p Vᵀ of Muon's momentum, with p = −0.3. It is computed either by exact SVD or by LC_K: a weighted sum of K Newton–Schulz iterates, each step with its own coefficients, starting from N/‖N‖_S8. The fit is `code/scripts/lc/` (branches two-track, neg-p-ce).
- **LC5m** (started 2026-11-27): LC5 with a hard constraint that every σ in (0, 1] lands between Muon's value and σ^p. Subject to that, it gets as close to σ^p as possible. It never gives a direction less than Muon does.
- **Results** (CE, lr 0.01, d0–d2; `results/dose_negative_p_ce.pdf` on neg-p-ce):

| | exact | LC12 | LC5 | Muon | AdamW |
|---|---|---|---|---|---|
| final val loss | 1.3027 | 1.3020 | 1.3109 | 1.3086 | 1.343 |
| ms/step (L40S) | 1081 | 159 | 123 | 117 | 107 |

- **Steps to Muon's final loss:** LC12 needs 6–13% fewer (mean ≈ 9%), exact ≈ 8% fewer. To reach AdamW's final loss (1.3391), exact needs only 2% fewer. These are step counts inside a fixed 10k-step schedule. In GPU time nothing has beaten Muon yet.

## Prior work

| Source | What it does | Difference from LC5m |
|---|---|---|
| Srivastava, "Heterogeneous Iteration for Spectral Powers", May 2026, <https://varunneal.github.io/essays/spectral-powers> | A method essay for Muon-style optimizers (Muon is p = 0; Shampoo/SOAP use −1/2 and −1/4). Same construction as LC: per-step odd quintics, Σ w_k(p) X_k, p ∈ (−0.9, 0.9), at most 1.65% error for σ in [1e-3, 1]. | Spectral-norm start; weights are Chebyshev polynomials in p; small σ "sent toward 0" (no floor); no training runs. We designed LC independently; our coefficient sets are our own. |
| Nilin, "Contra-Muon and Soft-Muon", May 2026, <https://nilin.github.io/contra-muon-and-soft-muon/>; modded-nanogpt PR #291 | Soft-Muon: a linear combination of the iterates of one fixed polynomial, σ^p with p = 0.1, Schatten-4 start. Contra-Muon: (1+c)·Muon − c·X₀. | Soft-Muon gives small σ less than Muon. Contra-Muon boosts small σ, which acts as a floor, but it is never stated as a constraint (our inference). |
| DynMuon (Wu, Shah, Silwal, Zhang), arXiv 2605.17109 | Schedules p from 1 down to −0.25; negative p is computed as Muon × a Taylor approximation. Beats Muon: val 3.171 vs 3.190 at 127M. | No sum of iterates, no error guarantee. Fixed p = −0.25 for the whole run is "substantially worse" than the schedule; p ≤ −0.75 is unstable. |
| Freon (Shumaylov et al.), arXiv 2605.11181 | σ^(1−2c) with p = −1/3 or −1/2, computed by a rational QDWH/Zolotarev iteration. Best p in [−1, 0] on GPT-2 124M. | Not polynomial, no floor. Its numbers are only in figures. |
| SAMuon (Wu, Yu, Zhang, Woodland), arXiv 2608.25990 | Keeps every direction's scale in [1, γ] (never below Muon); the scale depends on the direction's singular-value rank. Uses a low-rank SVD (k ≈ 39–71), or power iteration in SAMuon-lite. 13–24% fewer tokens than Muon (124M–1B, e.g. 3.1953 → 3.1658 at 124M). | Closest in spirit, but pays for an SVD. Overhead at 1B: 7.4% (lite: 0.5%). One seed per configuration, results in tokens only; the wall-clock appendix was not read. |
| Muon^p (Dong & Sawin), arXiv 2606.13867 | p ∈ (0, 1) only. | No negative p. |

Not read: SMuon (Massena et al.), HTMuon (2603.10067) beyond its abstract, and Gawlik & Nakatsukasa on composite p-th roots.

## Takeaways

- Negative p and sums of iterates are both already published. The Muon floor (LC5m's hard "between Muon and σ^p" constraint) was found nowhere.
- LC5m costs Muon's 15 matrix products, with no SVD.
- Every source reports gains in tokens or steps. None reports a wall-clock win for p < 0, and our main metric is real time or FLOPs.
- Why SAMuon isn't widely used yet (our guess): it is about 3 months old, uses one seed, goes up to 1B only, reports tokens rather than time, and its SVD is awkward with sharded weights.
