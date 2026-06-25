"""Assess the 3-seed 19-dataset benchmark for skill-learnability: AUBC per cell, per-dataset method
rankings + spread, cross-seed reproducibility, gain-over-Random, Friedman differentiation, and whether
the best method VARIES by dataset in a learnable way (signal vs noise)."""
import numpy as np
from scipy.stats import friedmanchisquare, rankdata
from medal_bench.analysis.derived import load_curves, aubc

DS19 = ["btcv_synapse","flare22","mmwhs_ct","hvsmr2016","ext_abdoment1k","ext_brats2020",
        "msd_task07_pancreas","isic2018","care_leftatrium_2026","kits19","msd_task03_liver",
        "msd_task04_hippocampus","refuge","glas2015","msd_task09_spleen","origa","kvasir_seg","liqa_mri","busi"]
M = ["P0","P1","P2","P3","P4","P5","P6","P7","P8","P9"]
NAME = {"P0":"Random","P1":"Entropy","P2":"BALD","P3":"CoreSet","P4":"BADGE","P5":"Ent+Core",
        "P6":"SelUnc","P7":"SAM-Core","P8":"TypiClust","P9":"PAAL"}
SEEDS = [1000,2000,3000]
curves = load_curves("runs/frozen_v5")

A = {}  # (ds,m,seed)->aubc
for (ds,m,s),c in curves.items():
    if ds in DS19 and m in M and s in SEEDS and c:
        fr=[x for x,_ in c]; sc=[y for _,y in c]
        if len(fr)>=2: A[(ds,m,s)] = aubc(fr,sc)
dm = {}  # (ds,m)->(mean,std over seeds)
for ds in DS19:
    for m in M:
        v=[A[(ds,m,s)] for s in SEEDS if (ds,m,s) in A]
        if v: dm[(ds,m)]=(float(np.mean(v)),float(np.std(v)),len(v))

print("=== PER-DATASET (AUBC mean over 3 seeds; best vs Random) ===")
bestcnt={m:0 for m in M}; spreads=[]; p0rank=[]; toprank_std=[]
for ds in DS19:
    row=[(m,dm[(ds,m)][0]) for m in M if (ds,m) in dm]
    row.sort(key=lambda x:-x[1]); best=row[0]; spread=best[1]-row[-1][1]
    p0=dm[(ds,"P0")][0]; pr=1+sum(1 for _,v in row if v>p0)
    bestcnt[best[0]]+=1; spreads.append(spread); p0rank.append(pr)
    # cross-seed std of the top method (reproducibility of the winner)
    toprank_std.append(dm[(ds,best[0])][1])
    print(f"{ds:24s} best={NAME[best[0]]:9s}{best[1]:.3f}  Random={p0:.3f}(rank{pr})  spread={spread:.4f}")

print("\n=== PER-METHOD (over 19 datasets) ===")
# avg rank per dataset
rank_acc={m:[] for m in M}
for ds in DS19:
    vals=np.array([dm[(ds,m)][0] for m in M]); rk=len(M)+1-rankdata(vals)  # rank1=best
    for i,m in enumerate(M): rank_acc[m].append(rk[i])
print(f"{'method':12s} {'avgAUBC':>8s} {'avgRank':>8s} {'wins':>5s} {'gainVsRandom':>12s} {'xseed_std':>9s}")
for m in M:
    vals=[dm[(ds,m)][0] for ds in DS19]; stds=[dm[(ds,m)][1] for ds in DS19]
    gains=[dm[(ds,m)][0]-dm[(ds,"P0")][0] for ds in DS19]
    print(f"{NAME[m]+'('+m+')':12s} {np.mean(vals):8.4f} {np.mean(rank_acc[m]):8.2f} {bestcnt[m]:5d} {np.mean(gains):+12.4f} {np.mean(stds):9.4f}")

print("\n=== DIFFERENTIATION & LEARNABILITY ===")
print(f"median per-dataset AUBC spread (best-worst): {np.median(spreads):.4f}  (range {min(spreads):.4f}-{max(spreads):.4f})")
print(f"Random's mean rank across datasets: {np.mean(p0rank):.1f}/10  (1=Random-always-best, 10=always-worst)")
print(f"best-method distribution: {{{', '.join(f'{NAME[m]}:{c}' for m,c in bestcnt.items() if c>0)}}}")
print(f"mean cross-seed std of the winning method: {np.mean(toprank_std):.4f}  (low=reproducible winners)")
tbl=np.array([[dm[(ds,m)][0] for m in M] for ds in DS19])
st,p=friedmanchisquare(*[tbl[:,i] for i in range(len(M))])
print(f"Friedman (any method differs?): chi2={st:.1f} p={p:.2e}")
# how much does the BEST method beat the 2nd-best (margin) -- skill value
margins=[]
for ds in DS19:
    v=sorted([dm[(ds,m)][0] for m in M],reverse=True); margins.append(v[0]-v[1])
print(f"median best-vs-2nd margin: {np.median(margins):.4f}  (how much picking THE best beats picking 2nd)")
print(f"median best-vs-Random margin: {np.median([dm[(ds,row[0][0])][0]-dm[(ds,'P0')][0] for ds in DS19 for row in [sorted([(m,dm[(ds,m)][0]) for m in M],key=lambda x:-x[1])]]):.4f}")
