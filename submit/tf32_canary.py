"""Cross-arch numerical canary: does TF32-off seeding make the trained nnU-Net
byte-comparable across GPU archs? Trains the real model on a FIXED synthetic
512px batch and prints a high-precision fingerprint. Run on V100/A40/H100 and diff.
"""
import torch, numpy as np
from medal_bench.runner.seeds import seed_all
from medal_bench.models.nnunet import build_unet_2d

seed_all(1000)
dev = "cuda:0"
name = torch.cuda.get_device_name(0)

# fixed synthetic batch (seeded, identical on every host)
rng = np.random.RandomState(1000)
B, C, H, W, NC = 4, 1, 512, 512, 2
x = torch.from_numpy(rng.randn(B, C, H, W).astype(np.float32)).to(dev)
y = torch.from_numpy((rng.rand(B, H, W) > 0.7).astype(np.int64)).to(dev)
xe = torch.from_numpy(rng.randn(2, C, H, W).astype(np.float32)).to(dev)  # eval probe

model = build_unet_2d(input_channels=C, num_classes=NC, dropout_p=0.1).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
lossf = torch.nn.CrossEntropyLoss()
model.train()
losses = []
for it in range(40):
    opt.zero_grad()
    out = model(x)
    out = out[0] if isinstance(out, (list, tuple)) else out  # deep-sup -> highest res
    loss = lossf(out, y)
    loss.backward()
    opt.step()
    losses.append(float(loss.item()))

model.eval()
with torch.no_grad():
    ev = model(xe)
    ev = ev[0] if isinstance(ev, (list, tuple)) else ev
    ev = ev.detach().double().cpu()

print(f"GPU={name}")
print(f"tf32_cudnn={torch.backends.cudnn.allow_tf32} tf32_matmul={torch.backends.cuda.matmul.allow_tf32}")
print(f"final_loss={losses[-1]:.10f}")
print(f"loss_5={losses[5]:.10f} loss_20={losses[20]:.10f}")
print(f"eval_logit_sum={ev.sum().item():.8f}")
print(f"eval_logit_absmax={ev.abs().max().item():.8f}")
print(f"eval_logit_mean={ev.mean().item():.10f}")
print(f"eval_softmax_fg_sum={torch.softmax(ev,dim=1)[:,1].sum().item():.8f}")
