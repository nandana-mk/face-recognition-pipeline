"""
23_train_adapter.py — train the degradation-invariant adapter.

    python 23_train_adapter.py

FIXED vs the previous version (which made things WORSE: 0.494 -> 0.437):

  BUG 1 - the adapter did not start as the identity function.
     `x + MLP(x)` is only identity when MLP(x) ~ 0, but with BatchNorm the branch starts at unit
     scale, so epoch 1 already destroyed the embedding (val cos 0.344 vs 0.494 baseline) and the
     rest of training just crawled back. FIX: zero-init the last layer -> the residual branch
     outputs exactly 0, so the adapter begins as a PERFECT identity and can only improve.

  BUG 2 - the loss optimised alignment, not discriminability.
     `1 - cos(A(deg), ref)` is minimised by collapsing every embedding toward a common point,
     which raises similarity to the right person AND to everyone else - destroying precision.
     FIX: InfoNCE. Pull each probe toward its OWN reference and push it AWAY from the other
     references in the batch. That is the recognition task itself, and collapse cannot game it.

Metric reported is TOP-1 IDENTIFICATION ACCURACY over held-out identities (does the probe
retrieve its own reference out of all val references?) - mean cosine does not tell you whether
recognition actually improved.

Output: models/adapter_deg.pt + training curve.
"""
import os, datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

PAIRS   = r"C:\dev\train_val\models\pairs.npz"
OUT_PT  = r"C:\dev\train_val\models\adapter_deg.pt"
PLOTDIR = r"C:\dev\train_val\reports\plots"
VAL_FRAC, EPOCHS, BATCH, LR, HIDDEN, PATIENCE, SEED = 0.15, 60, 512, 3e-4, 1024, 10, 0
TEMP   = 0.05      # InfoNCE temperature for L2-normalised embeddings
LAMBDA = 1.0       # weight on the clean-preservation term

torch.manual_seed(SEED); np.random.seed(SEED)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] device: {dev}")

d = np.load(PAIRS)
DEG, TGT, PID = d['deg'], d['tgt'], d['pid']
CLEAN, CPID = d['clean'], d['clean_pid']
P = CLEAN.shape[0]
print(f"[INFO] {P} identities, {DEG.shape[0]} (probe, reference) pairs")

# split BY PERSON
perm = np.random.permutation(P)
n_val = max(1, int(P * VAL_FRAC))
val_person = np.zeros(P, bool); val_person[perm[:n_val]] = True
tr_mask, va_mask = ~val_person[PID], val_person[PID]
print(f"[INFO] train {P-n_val} identities ({tr_mask.sum()} pairs) | "
      f"val {n_val} identities ({va_mask.sum()} pairs)")

Xtr = torch.from_numpy(DEG[tr_mask]).to(dev); Ytr = torch.from_numpy(TGT[tr_mask]).to(dev)
Ptr = torch.from_numpy(PID[tr_mask].astype(np.int64)).to(dev)
Xva = torch.from_numpy(DEG[va_mask]).to(dev)
Ctr = torch.from_numpy(CLEAN[~val_person[CPID]]).to(dev)
Cva = torch.from_numpy(CLEAN[val_person[CPID]]).to(dev)

# ---- val identification setup: one reference row per val identity ----
val_pids = CPID[val_person[CPID]]
Rva = torch.from_numpy(CLEAN[val_person[CPID]]).to(dev)          # (V,512) references
pid2row = {int(p): i for i, p in enumerate(val_pids)}
va_label = torch.tensor([pid2row[int(p)] for p in PID[va_mask]], dtype=torch.long, device=dev)
print(f"[INFO] val gallery: {Rva.shape[0]} identities, {Xva.shape[0]} probes\n")

def identify(Z, R):
    """top-1 accuracy: does each probe retrieve its own reference?"""
    return ((Z @ R.t()).argmax(1) == va_label).float().mean().item()

base_cos = (Xva * Rva[va_label]).sum(1).mean().item()
base_acc = identify(Xva, Rva)
print(f"[BASE] val mean cos {base_cos:.3f}   TOP-1 IDENTIFICATION {base_acc:.1%}\n")

class Adapter(nn.Module):
    def __init__(self, dim=512, h=HIDDEN):
        super().__init__()
        self.fc1 = nn.Linear(dim, h); self.act = nn.GELU(); self.fc2 = nn.Linear(h, dim)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)   # EXACT identity at init
    def forward(self, x):
        return F.normalize(x + self.fc2(self.act(self.fc1(x))), dim=1)

model = Adapter().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
print(f"[INFO] adapter params: {sum(p.numel() for p in model.parameters()):,}")
with torch.no_grad():
    model.eval(); chk = identify(model(Xva), model(Rva)); model.train()
print(f"[INFO] sanity: top-1 at init = {chk:.1%} (must equal baseline {base_acc:.1%})\n")

def evaluate():
    model.eval()
    with torch.no_grad():
        Z = model(Xva); R = model(Rva)
        acc = identify(Z, R)
        cos = (Z * R[va_label]).sum(1).mean().item()
        pres = (model(Cva) * Cva).sum(1).mean().item()
    model.train()
    return acc, cos, pres

hist = {'ep': [], 'loss': [], 'acc': [], 'cos': [], 'pres': []}
best, best_ep, bad = base_acc, 0, 0
ntr = Xtr.shape[0]
for ep in range(1, EPOCHS + 1):
    idx = torch.randperm(ntr, device=dev)
    tot, nb = 0.0, 0
    for i in range(0, ntr, BATCH):
        b = idx[i:i+BATCH]
        if b.numel() < 8: continue
        z = model(Xtr[b]); t = Ytr[b]; pid = Ptr[b]
        logits = (z @ t.t()) / TEMP
        same = (pid[:, None] == pid[None, :])            # other pairs of the SAME person
        eye = torch.eye(len(b), dtype=torch.bool, device=dev)
        logits = logits.masked_fill(same & ~eye, -1e4)   # they are not negatives
        loss_nce = F.cross_entropy(logits, torch.arange(len(b), device=dev))
        cb = Ctr[torch.randint(0, Ctr.shape[0], (min(BATCH, Ctr.shape[0]),), device=dev)]
        loss_pres = (1 - (model(cb) * cb).sum(1)).mean()
        loss = loss_nce + LAMBDA * loss_pres
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); nb += 1
    sched.step()
    acc, cos, pres = evaluate()
    hist['ep'].append(ep); hist['loss'].append(tot/max(nb,1))
    hist['acc'].append(acc); hist['cos'].append(cos); hist['pres'].append(pres)
    flag = ""
    if acc > best:
        best, best_ep, bad = acc, ep, 0
        torch.save({'state_dict': model.state_dict(), 'hidden': HIDDEN, 'dim': 512}, OUT_PT)
        flag = "  <- saved"
    else:
        bad += 1
    print(f"  ep {ep:3d}  loss {tot/max(nb,1):.4f}  TOP-1 {acc:.1%} (base {base_acc:.1%})  "
          f"cos {cos:.3f}  clean-preserve {pres:.4f}{flag}")
    if bad >= PATIENCE:
        print(f"  early stop (no improvement for {PATIENCE} epochs)"); break

print("\n" + "=" * 68)
print("RESULT (identities never seen in training)")
print("=" * 68)
print(f"  TOP-1 identification   {base_acc:.1%}  ->  {best:.1%}   ({(best-base_acc)*100:+.1f} pts)")
if best_ep > 0:
    print(f"  mean cos               {base_cos:.3f}  ->  {hist['cos'][best_ep-1]:.3f}")
    print(f"  clean preservation     {hist['pres'][best_ep-1]:.4f}   (1.000 = clean untouched)")
if best <= base_acc + 1e-9:
    print("\n  >>> NO IMPROVEMENT. The adapter never beat the identity function.")
    print("      Most likely cause: not enough training identities (LFW gives only ~1.4k here).")
    print("      CASIA-WebFace (10.5k IDs) is the next thing to try - not more epochs.")
else:
    print(f"\n  [SAVED] {OUT_PT}  (best epoch {best_ep})")
    print("  Now run: python 24_eval_adapter.py <lfw folder>   <-- the real gates")

os.makedirs(PLOTDIR, exist_ok=True)
pdir = os.path.join(PLOTDIR, datetime.datetime.now().strftime("adapter_%H%M%S"))
os.makedirs(pdir, exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].plot(hist['ep'], hist['loss'], color='#C0392B'); ax[0].set_title('training loss')
ax[0].set_xlabel('epoch'); ax[0].grid(alpha=.3)
ax[1].plot(hist['ep'], hist['acc'], 'o-', color='#0C7C88', label='top-1 (adapted)')
ax[1].axhline(base_acc, ls='--', color='grey', label=f'baseline {base_acc:.1%}')
ax[1].plot(hist['ep'], hist['pres'], 's--', color='#B67611', label='clean preservation')
ax[1].set_xlabel('epoch'); ax[1].set_title('held-out identification accuracy')
ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(pdir, 'adapter_training.png'), dpi=150); plt.close(fig)
print(f"[SAVED] plot -> {pdir}")
