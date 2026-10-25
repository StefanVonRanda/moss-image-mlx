"""Goodhart probe: can the stamping score be driven down by optimising the pixels against it directly?

PGD on a stamped image under an L-inf budget (eps/255), minimising a soft version of the score computed by the same
network the localiser uses (DINOv2-small layer 6, the 4 phase tilings, tile borders excluded, textured tokens):
    loss = mean_i softplus((smax_i - 0.75) / 0.02),  smax_i = soft-max over tokens >= 56 px away of cos(f_i, f_j)
Then everything is re-measured with evaluate.measure: the attacked network, a held-out network (DINOv2-base) and the
pixel-domain NCC repetition. Usage: goodhart.py IMAGE OUT_PREFIX [eps] [steps]
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from stampscore.core import Extractor, load, PHASES, TILE, PATCH, MEAN, STD
from evaluate import measure


def token_feats(model, x, dev, layer=6):
    """x: (3,H,W) float 0..1 with grad. Returns normalised tokens, centres, textured+non-border mask."""
    feats, pos, keep = [], [], []
    g = TILE // PATCH
    lum = x.detach().mean(0)
    for dy, dx in PHASES["p4"]:
        sub = x[:, dy:, dx:]
        h, w = sub.shape[1:]
        ys = list(range(0, h - TILE + 1, TILE)) + ([h - TILE] if h % TILE else [])
        xs = list(range(0, w - TILE + 1, TILE)) + ([w - TILE] if w % TILE else [])
        tiles = torch.stack([sub[:, y:y + TILE, xx:xx + TILE] for y in ys for xx in xs])
        out = model(pixel_values=(tiles - MEAN.to(dev)) / STD.to(dev), output_hidden_states=True)
        f = out.hidden_states[layer][:, 1:].reshape(-1, out.hidden_states[layer].shape[-1])
        P, E = [], []
        for y in ys:
            for xx in xs:
                for k in range(g * g):
                    r, c = divmod(k, g)
                    P.append((y + r * PATCH + 7 + dy, xx + c * PATCH + 7 + dx))
                    E.append(min(r, c, g - 1 - r, g - 1 - c) < 1)
        P, E = np.array(P, float), np.array(E)
        _, idx = np.unique(P, axis=0, return_index=True)
        sd = np.array([float(lum[int(cy) - 7:int(cy) + 7, int(cx) - 7:int(cx) + 7].std()) for cy, cx in P[idx]])
        feats.append(f[torch.from_numpy(idx).to(dev)]); pos.append(P[idx]); keep.append((sd > .02) & ~E[idx])
    f = torch.cat(feats)
    return Fn.normalize(f, dim=1), np.concatenate(pos), np.concatenate(keep)


def main(path, prefix, eps=4.0, steps=60):
    torch.set_grad_enabled(True)  # stampscore.core and tiles switch autograd off at import
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    ex = Extractor(device=dev)
    for p_ in ex.model.parameters():
        p_.requires_grad_(False)
    img = load(path)
    x0 = torch.from_numpy(img).permute(2, 0, 1).float().to(dev) / 255
    with torch.no_grad():
        _, pos, keep = token_feats(ex.model, x0, dev)
    pt = torch.from_numpy(pos[keep]).float().to(dev)
    far = torch.cdist(pt, pt) >= 56
    ki = torch.from_numpy(np.where(keep)[0]).to(dev)
    delta = torch.zeros_like(x0, requires_grad=True)
    e, a = eps / 255, eps / 255 / 10
    hist = []
    for it in range(steps):
        f, _, _ = token_feats(ex.model, (x0 + delta).clamp(0, 1), dev)
        f = f[ki]
        S = (f @ f.T).masked_fill(~far, -1)
        smax = torch.logsumexp(S / .01, 1) * .01
        loss = Fn.softplus((smax - .75) / .02).mean()
        loss.backward()
        with torch.no_grad():
            delta -= a * delta.grad.sign()
            delta.clamp_(-e, e)
            delta.grad.zero_()
        hist.append((it, float(loss), float((S.max(1).values >= .80).float().mean())))
        if it % 10 == 0:
            print(hist[-1], flush=True)
    adv = ((x0 + delta).clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(adv).save(prefix + "_adv.png")
    torch.set_grad_enabled(False)
    mask = np.ones(img.shape[:2])
    r = measure(img, adv, mask)
    r["eps"], r["steps"], r["hist"] = eps, steps, hist
    Path(prefix + "_adv.json").write_text(json.dumps(r, indent=1))
    print(json.dumps({k: v for k, v in r.items() if k != "hist"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(float(a) for a in sys.argv[3:4]), *(int(a) for a in sys.argv[4:5]))
