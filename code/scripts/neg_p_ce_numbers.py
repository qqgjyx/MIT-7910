"""Numbers of results/dose_negative_p_ce.tex that dose.txt and two_track_ce.txt do not print: the length of LC12 at
Muon's GPU time (from the CE timing pilot), the spread of Muon's final loss over data orders, and the JS to Muon at
equal step and at equal training loss (as in dose_paths_ce.pdf). From the code dir, after scripts/dose_ce.sh:

    python scripts/neg_p_ce_numbers.py
"""
from itertools import combinations

import numpy as np

from optsplit.analysis.distance import distance
from optsplit.analysis.dose import equal_loss
from optsplit.analysis.io import Run
from optsplit.analysis.two_track import step_time

t = {m: np.mean([step_time(Run(f'ts10m_ce_tt_pilot/{m}_d{d}')) for d in (0, 1)]) for m in ('muon', 'lc12')}
r = t['muon'] / t['lc12']
print(f'ms per step: Muon {t["muon"] * 1e3:.1f}, LC12 {t["lc12"] * 1e3:.1f}; LC12 at equal GPU time: '
      f'{round(10000 * r)} steps, warmup {round(200 * r)}')

muon = {d: Run(f'ts10m_main/muon_d{d}') for d in range(5)}
val = [muon[d].series('eval', 'val_loss')[1][-1] for d in muon]
print(f'Muon final val. loss, data orders 0-4: mean {np.mean(val):.4f}, standard deviation {np.std(val, ddof=1):.4f}')

steps, levels = [500, 1000, 2000, 5000, 10000], [3, 2.5, 2, 1.81, 1.6, 1.5, 1.4, 1.35]
print(f'\nJS (bits) at steps {steps} | at training loss {levels}')


def line(name, a, b):
    c = equal_loss(a, b)
    eq = [np.interp(-x, -c['loss'], c['js'], left=np.nan, right=np.nan) for x in levels]
    st = [distance(a, s, b, s)[0] for s in steps]
    print(f'{name:22s}', ' '.join(f'{x:.4f}' for x in st), ' | ', ' '.join(f'{x:.4f}' for x in eq))


line('AdamW vs Muon', Run('ts10m_main/adamw_d0'), muon[0])
for p in ['-1', '-0.75', '-0.5', '-0.3', '-0.1', '0', '0.25', '0.5', '0.75']:
    line(f'p={p} vs Muon', Run(f'ts10m_ce_svp/muon_p{p}_lr0.01_d0'), muon[0])
for a, b in combinations(range(5), 2):
    line(f'Muon d{a} vs d{b}', muon[a], muon[b])
m3 = {d: Run(f'ts10m_ce_svp/muon_p-0.3_lr0.01_d{d}') for d in range(3)}
for a, b in combinations(range(3), 2):
    line(f'p=-0.3 d{a} vs d{b}', m3[a], m3[b])
