"""Final validation loss of every method of the slides, and the step at which its validation loss first reaches
AdamW's final one (same init and data order), per replicate: the inits 0 (the slides' runs), 1 and 2 on data order 0
(scripts/init_seeds_submit.sh), or with --orders the data orders 0-4 of init 0. Last column: the mean final loss over
the replicates that have the run. Run by scripts/init_seeds_eval.sh:

    python scripts/init_seeds_numbers.py [--orders]
"""
import sys

import numpy as np

from optsplit.analysis.dose import first_step
from optsplit.analysis.io import load_runs


def methods(i, d):
    """{method: run} of init i, data order d."""
    if i == 0:
        g = dict(AdamW=f'ts10m_main/adamw_d{d}', Muon=f'ts10m_main/muon_d{d}', smoothed=f'ts10m_ce_smooth/smuon_d{d}',
                 svp='ts10m_ce_svp', gram=f'ts10m_ce_gram*/*gram_d{d}', lc=f'ts10m_ce_tt/muon_p-0.3_lc{{}}_d{d}')
    else:
        g = dict(AdamW=f'ts10m_traj_i{i}/adamw_d{d}', Muon=f'ts10m_traj_i{i}/muon_d{d}',
                 smoothed=f'ts10m_ce_smooth_i{i}/smuon_d{d}', svp=f'ts10m_ce_svp_i{i}',
                 gram=f'ts10m_ce_lc_i{i}/muon_p-0.3_lcgram_d{d}', lc=f'ts10m_ce_lc_i{i}/muon_p-0.3_lc{{}}_d{d}')
    one = lambda pat: (load_runs(pat) or [None])[0]
    out = {k: one(g[k]) for k in ('AdamW', 'Muon')}
    out['smoothed Muon'] = one(g['smoothed'])
    svd = load_runs(f"{g['svp']}/muon_p*_lr0.01_d{d}")
    out |= {f'exact SVD p={r.cfg.optim.muon.sv_power:g}': r for r in sorted(svd, key=lambda r: r.cfg.optim.muon.sv_power)}
    out |= {f'LC{k}': one(g['lc'].format(k)) for k in (12, 8, 5)}
    out['Gram LC'] = one(g['gram'])
    return out


def main():
    orders = '--orders' in sys.argv
    reps = {f'd{d}': methods(0, d) for d in range(5)} if orders else {f'init {i}': methods(i, 0) for i in (0, 1, 2)}
    names = list(dict.fromkeys(k for m in reps.values() for k in m))
    print('method | final validation loss (step to AdamW\'s final validation loss): ' + ' | '.join(reps) + ' | mean')
    for k in names:
        cells, finals = [], []
        for m in reps.values():
            r, a = m.get(k), m['AdamW']
            if r is None or not r.records_of('eval'):
                cells.append('-')
                continue
            final = r.series('eval', 'val_loss')[1][-1]
            step = first_step(r, 'val_loss', a.series('eval', 'val_loss')[1][-1]) if a else float('nan')
            cells.append(f'{final:.4f} ({step:.0f}){"" if r.done else " running"}')
            finals.append(final)
        print(f'{k} | ' + ' | '.join(cells) + f' | {np.mean(finals) if finals else float("nan"):.4f} (n={len(finals)})')


if __name__ == '__main__':
    main()
