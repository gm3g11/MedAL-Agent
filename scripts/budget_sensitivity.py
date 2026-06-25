"""Decisive test for the P9/PAAL-too-low claim: does P9 catch up at higher budget, or is it last
at every budget point? Per round-index (budget progression) method mean DSC + P9/BADGE rank+gain."""
import numpy as np
from collections import defaultdict
from medal_bench.analysis.derived import load_curves
DS19=["btcv_synapse","flare22","mmwhs_ct","hvsmr2016","ext_abdoment1k","ext_brats2020","msd_task07_pancreas",
 "isic2018","care_leftatrium_2026","kits19","msd_task03_liver","msd_task04_hippocampus","refuge","glas2015",
 "msd_task09_spleen","origa","kvasir_seg","liqa_mri","busi"]
M=["P0","P1","P2","P3","P4","P5","P6","P7","P8","P9"]; SEEDS=[1000,2000,3000]
NM={"P0":"Random","P1":"Entropy","P2":"BALD","P3":"CoreSet","P4":"BADGE","P5":"Ent+Core","P6":"SelUnc","P7":"SAM-Core","P8":"TypiClust","P9":"PAAL"}
cur=load_curves("results/frozen_v5")
cells=defaultdict(dict)
for (ds,m,s),c in cur.items():
    if ds in DS19 and m in M and s in SEEDS: cells[(ds,s)][m]=[d for _,d in c]
maxr=max((len(v) for cell in cells.values() for v in cell.values()), default=0)
print("=== per budget-round: P9/PAAL & BADGE rank + gain-vs-Random (cells where all 10 methods reached that round) ===")
print(f"{'round':>5s} {'n':>4s} | {'P9rank':>6s} {'P9dsc':>6s} {'P9-Rnd':>7s} | {'BADGErk':>7s} {'BADGE-Rnd':>9s} | best")
for ri in range(maxr):
    vals={m:[] for m in M}
    for (ds,s),mc in cells.items():
        if len(mc)==10 and all(len(mc[m])>ri for m in M):
            for m in M: vals[m].append(mc[m][ri])
    if len(vals["P0"])<20: continue
    mu={m:float(np.mean(vals[m])) for m in M}
    rk=sorted(M,key=lambda m:-mu[m])
    print(f"{ri:>5d} {len(vals['P0']):>4d} | {rk.index('P9')+1:>6d} {mu['P9']:>6.3f} {mu['P9']-mu['P0']:>+7.3f} | {rk.index('P4')+1:>7d} {mu['P4']-mu['P0']:>+9.3f} | {NM[rk[0]]}")
print("\n=== at each cell's FINAL (highest) budget: method mean DSC + rank ===")
fin={m:[] for m in M}
for (ds,s),mc in cells.items():
    if len(mc)==10:
        for m in M: fin[m].append(mc[m][-1])
mu={m:float(np.mean(fin[m])) for m in M}; rk=sorted(M,key=lambda m:-mu[m])
for i,m in enumerate(rk): print(f"  {i+1}. {NM[m]:9s} {mu[m]:.3f} ({mu[m]-mu['P0']:+.3f} vs Random)")
