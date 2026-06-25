"""P6 collapse forensic (audit #3). Freeze CARE-LA P6 round-4/5 labeled sets, retrain
each with 3 training seeds; P0 round-5 as a same-trainer control. Characterize the
selected slices (unique cases, adjacent-slice redundancy, GT fg-size, padding).
Run on V100 (the arch care_LA P6 originally used) so the seed_base=1000 case self-checks.
"""
import json, glob, os
import numpy as np, torch
from collections import defaultdict
from medal_bench.runner.run_one import _build_adapter, DEFAULT_DATA_ROOT, DEFAULT_FOUND_CACHE
from medal_bench.runner.splits import make_split
from medal_bench.profiles import build_run_config
from medal_bench.runner.al_loop import _load_or_make_pool_indices, SplitView, _IndexedSubset, _build_model
from medal_bench.runner.trainer import train_from_scratch
from medal_bench.runner.eval import eval_segmentation
from medal_bench.runner.seeds import seed_all, seed_torch, component_seeds

DS, SEED, DEV = "care_leftatrium_2026", 1000, "cuda:0"
print(f"GPU={torch.cuda.get_device_name(0)}  tf32_cudnn={torch.backends.cudnn.allow_tf32}")

def load_frozen(pol):
    jf = (glob.glob(f"runs/stage2_full/{DS}__{pol}__s{SEED}.jsonl")
          + glob.glob(f"runs/stage2_wave2/{DS}__{pol}__s{SEED}.jsonl"))
    recs = [json.loads(l) for l in open(jf[0])]
    init_ids = recs[0]["initial_labeled_ids"]
    sel = [r.get("selected_ids", []) for r in recs]
    def at(r):                       # labeled set at round r = init ∪ selected[0..r-1]
        s = set(init_ids)
        for k in range(r): s.update(sel[k])
        return sorted(s)
    odsc = [r["metrics"]["mean_dsc_fg"] for r in recs]
    return at, odsc

p6_at, p6_dsc = load_frozen("P6")
p0_at, p0_dsc = load_frozen("P0")
print(f"orig P6 dsc-by-round: {[round(x,3) for x in p6_dsc]}")
print(f"orig P0 dsc-by-round: {[round(x,3) for x in p0_dsc]}")

# build adapter/split/pool/val exactly like run_al
adapter = _build_adapter(DS, DEFAULT_DATA_ROOT)
seed_all(SEED)
split = make_split(adapter, seed=SEED)
train_view = SplitView(adapter, split.train, "train")
val_view = SplitView(adapter, split.val, "val")
PRE = os.path.join(os.path.dirname(DEFAULT_FOUND_CACHE), "preprocessed")
bc = dict(profile_name="bench512_v4", policy_id="P6", policy_config={}, dataset_name=DS,
          seed=SEED, out_jsonl="/tmp/p6f.jsonl", device=DEV, foundation_features_fn=None,
          num_classes=adapter.num_classes, preproc_cache_dir=PRE)
cfg = build_run_config(pool_size=len(train_view), **bc)
pool_idx = _load_or_make_pool_indices(cfg, adapter.name, train_view)
cfg = build_run_config(pool_size=len(pool_idx), **bc)
rng = np.random.RandomState(SEED)
val_idx = list(range(len(val_view)))
if cfg.val_cap and len(val_idx) > cfg.val_cap:
    val_idx = rng.choice(val_idx, size=cfg.val_cap, replace=False).tolist()
pool_subset = _IndexedSubset(train_view, pool_idx, cfg.train.image_size, cfg.train.aspect_preserve, cache_dir=PRE)
val_subset = _IndexedSubset(val_view, val_idx, cfg.train.image_size, cfg.train.aspect_preserve, cache_dir=PRE)
NC = adapter.num_classes
IC = int(pool_subset[0].image.shape[0])
id2local = {pool_subset[i].sample_id: i for i in range(len(pool_subset))}
print(f"pool_N={len(pool_subset)} val_N={len(val_subset)} num_classes={NC}")

def characterize(name, ids):
    local = [id2local[s] for s in ids if s in id2local]
    cases = defaultdict(list); fg = []
    for li in local:
        s = pool_subset[li]
        fg.append(int((np.asarray(s.mask) > 0).sum()))
        pid = s.patient_id or s.sample_id
        cases[pid].append(s.slice_index if s.slice_index is not None else -1)
    adj = tot = 0
    for sls in cases.values():
        sl = sorted(x for x in sls if x >= 0)
        for a, b in zip(sl, sl[1:]):
            tot += 1; adj += (b - a == 1)
    fg = np.array(fg); HW = cfg.train.image_size ** 2
    print(f"  [{name}] n={len(local)} unique_cases={len(cases)} "
          f"fg=0 slices={int((fg==0).sum())}({100*(fg==0).mean():.0f}%) "
          f"fg_median={int(np.median(fg))}px({100*np.median(fg)/HW:.2f}%) fg_mean={fg.mean():.0f} "
          f"adjacent_slices={adj}/{tot}({100*adj/max(1,tot):.0f}%)")

print("\n=== labeled-set characterization ===")
characterize("P6 r4", p6_at(4)); characterize("P6 r5", p6_at(5)); characterize("P0 r5", p0_at(5))

def retrain(tag, ids, r, seed_base):
    local = sorted({id2local[s] for s in ids if s in id2local})
    lds = _IndexedSubset(pool_subset, local, cfg.train.image_size, cfg.train.aspect_preserve)
    cs = component_seeds(seed_base + r); seed_all(seed_base + r); seed_torch(cs["model_init_seed"])
    m = _build_model(IC, NC, cfg.train).to(DEV)
    ts = train_from_scratch(m, lds, num_iters=cfg.train.num_iters, batch_size=cfg.train.batch_size,
        lr=cfg.train.lr, image_size=cfg.train.image_size, num_classes=NC, device=DEV,
        seed=cs["loader_seed"], dropout_seed=cs["dropout_seed"], adaptive=cfg.train.adaptive_iters,
        min_iters=cfg.train.min_iters, max_iters=cfg.train.max_iters, plateau_window=cfg.train.plateau_window,
        plateau_patience=cfg.train.plateau_patience, plateau_min_delta=cfg.train.plateau_min_delta,
        plateau_rel_delta=cfg.train.plateau_rel_delta)
    me = eval_segmentation(m, val_subset, num_classes=NC, image_size=cfg.train.image_size, device=DEV, compute_surface=False)
    print(f"  {tag} seed{seed_base} r{r}: DSC={me['mean_dsc_fg']:.4f} "
          f"detect={me['structure_detection_rate']:.3f} "
          f"stop_iter={ts.get('stop_iter')} stop={ts.get('stop_reason')} "
          f"last_loss={ts.get('last_loss',float('nan')):.4f} best_smooth={ts.get('best_smooth_loss',float('nan')):.4f}")

print("\n=== retrain frozen sets with 3 training seeds (seed1000 r4 should reproduce ~0.10) ===")
for sb in (1000, 2000, 3000):
    retrain("P6 r4", p6_at(4), 4, sb)
for sb in (1000, 2000, 3000):
    retrain("P6 r5", p6_at(5), 5, sb)
print("--- control: P0 labeled set, same trainer ---")
retrain("P0 r5", p0_at(5), 5, 1000)
print("DONE")
