import numpy as np
from medal_bench.analysis.derived import load_curves, aubc
DS19=["btcv_synapse","flare22","mmwhs_ct","hvsmr2016","ext_abdoment1k","ext_brats2020","msd_task07_pancreas",
 "isic2018","care_leftatrium_2026","kits19","msd_task03_liver","msd_task04_hippocampus","refuge","glas2015",
 "msd_task09_spleen","origa","kvasir_seg","liqa_mri","busi"]
M=["P0","P1","P2","P3","P4","P5","P6","P7","P8","P9"]; SEEDS=[1000,2000,3000]
NM={"P0":"Random","P4":"BADGE","P5":"Ent+Core","P8":"TypiClust","P2":"BALD","P7":"SAM-Core"}
cur=load_curves("runs/frozen_v5")
A={}
for (ds,m,s),c in cur.items():
    if ds in DS19 and m in M and s in SEEDS and len(c)>=2: A[(ds,m,s)]=aubc([x for x,_ in c],[y for _,y in c])

print("=== per-seed winner stability (is best-method real signal or noise?) ===")
stable=semi=0
for ds in DS19:
    bps=[max((m for m in M if (ds,m,s) in A),key=lambda m:A[(ds,m,s)]) for s in SEEDS]
    u=len(set(bps))
    if u==1: stable+=1
    elif u==2: semi+=1
    print(f"  {ds:22s} {bps} {'STABLE' if u==1 else '2of3' if u==2 else 'ALL-DIFFER'}")
print(f"-> same winner all 3 seeds: {stable}/19 | 2-of-3: {semi}/19 | all-differ: {19-stable-semi}/19")

dmean={(ds,m):np.mean([A[(ds,m,s)] for s in SEEDS if (ds,m,s) in A]) for ds in DS19 for m in M}
def pol(pick): return np.mean([A[(ds,pick(ds,s),s)] for ds in DS19 for s in SEEDS if (ds,pick(ds,s),s) in A])
rand=pol(lambda ds,s:"P0"); badge=pol(lambda ds,s:"P4")
orc=pol(lambda ds,s:max(M,key=lambda m:dmean[(ds,m)]))
orcps=pol(lambda ds,s:max((m for m in M if (ds,m,s) in A),key=lambda m:A[(ds,m,s)]))
GOOD=["P0","P4","P5","P8","P2","P7"]
bestgood=pol(lambda ds,s:max(GOOD,key=lambda m:dmean[(ds,m)]))
print("\n=== policy AUBC (mean over 19 ds x 3 seeds) — the skill ceiling ===")
print(f"always-Random            : {rand:.4f}")
print(f"always-BADGE (best fixed): {badge:.4f}  (+{badge-rand:.4f} vs Random)")
print(f"best-of-good-cluster/ds  : {bestgood:.4f}  (+{bestgood-badge:.4f} vs BADGE)")
print(f"ORACLE per-dataset-best  : {orc:.4f}  (+{orc-badge:.4f} vs BADGE, +{orc-rand:.4f} vs Random)")
print(f"ORACLE per-seed (cheat)  : {orcps:.4f}  (unachievable ceiling)")
# how often is BADGE within noise of the oracle?
within=sum(1 for ds in DS19 if dmean[(ds,max(M,key=lambda m:dmean[(ds,m)]))]-dmean[(ds,"P4")]<0.01)
print(f"\ndatasets where BADGE is within 0.01 AUBC of the per-dataset oracle: {within}/19")
