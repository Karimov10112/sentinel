"""Neyron tarmoqlar NumPy’da: DNN va autoenkoder (diplom, 2.4-band, A-ilova).

Arxitektura va o‘qitish sozlamalari A-ilovadagi PyTorch kodi bilan bir xil:
  DNN:         d → 256 → BN → ReLU → Dropout(0,3) → 128 → BN → ReLU → Dropout(0,3) → 64 → ReLU → k, Adam(lr=1e-3, wd=1e-5),
               paket 512, erta to‘xtatish (5 davr), CrossEntropy (ixtiyoriy sinf vaznlari)
  Autoenkoder: d → 64 → 16 → 8 → 16 → 64 → d (ReLU), Adam(lr=1e-3), MSE, faqat normal trafikda
PyTorch o‘rniga NumPy ishlatilgani uchun tizim Windows’da qo‘shimcha og‘ir kutubxonalarsiz ishlaydi.
"""
from __future__ import annotations

import numpy as np

F32 = np.float32


class Linear:
    def __init__(self, i: int, o: int, rng):
        bound = 1.0 / np.sqrt(i)                       # PyTorch nn.Linear standart initsializatsiyasi
        self.W = rng.uniform(-bound, bound, (i, o)).astype(F32)
        self.b = rng.uniform(-bound, bound, o).astype(F32)
        self.params = ["W", "b"]

    def forward(self, x, train=False):
        self.x = x
        return x @ self.W + self.b

    def backward(self, g):
        self.dW = self.x.T @ g
        self.db = g.sum(0)
        return g @ self.W.T


class BatchNorm:
    def __init__(self, n: int, momentum: float = 0.1, eps: float = 1e-5):
        self.gamma = np.ones(n, F32)
        self.beta = np.zeros(n, F32)
        self.rm = np.zeros(n, F32)
        self.rv = np.ones(n, F32)
        self.m, self.eps = momentum, eps
        self.params = ["gamma", "beta"]

    def forward(self, x, train=False):
        if train:
            mu, var = x.mean(0), x.var(0)
            n = x.shape[0]
            self.rm = (1 - self.m) * self.rm + self.m * mu
            self.rv = (1 - self.m) * self.rv + self.m * var * n / max(n - 1, 1)
            self.std = np.sqrt(var + self.eps)
            self.xh = (x - mu) / self.std
            return self.gamma * self.xh + self.beta
        return self.gamma * (x - self.rm) / np.sqrt(self.rv + self.eps) + self.beta

    def backward(self, g):
        n = g.shape[0]
        self.dgamma = (g * self.xh).sum(0)
        self.dbeta = g.sum(0)
        dxh = g * self.gamma
        return (n * dxh - dxh.sum(0) - self.xh * (dxh * self.xh).sum(0)) / (n * self.std)


class ReLU:
    params: list = []

    def forward(self, x, train=False):
        self.mask = x > 0
        return x * self.mask

    def backward(self, g):
        return g * self.mask


class Dropout:
    params: list = []

    def __init__(self, p: float, rng):
        self.p, self.rng = p, rng

    def forward(self, x, train=False):
        if not train or self.p == 0:
            return x
        self.mask = (self.rng.random(x.shape) >= self.p).astype(F32) / (1 - self.p)
        return x * self.mask

    def backward(self, g):
        return g * self.mask


class Net:
    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, x, train=False):
        for layer in self.layers:
            x = layer.forward(x, train)
        return x

    def backward(self, g):
        for layer in reversed(self.layers):
            g = layer.backward(g)

    def grads(self):
        for layer in self.layers:
            for p in layer.params:
                grad = getattr(layer, "d" + p)
                yield layer, p, grad

    # ------------------------------------------------------------ saqlash (npz) — kutubxona versiyasidan mustaqil
    def state(self) -> dict:
        st = {}
        for i, layer in enumerate(self.layers):
            for p in layer.params + (["rm", "rv"] if isinstance(layer, BatchNorm) else []):
                st[f"{i}.{p}"] = getattr(layer, p).copy()
        return st

    def load(self, st: dict) -> "Net":
        for k, v in st.items():
            i, p = k.split(".")
            setattr(self.layers[int(i)], p, np.asarray(v, F32))
        return self

    def predict(self, X, bs: int = 8192):
        return np.vstack([self.forward(X[i:i + bs]) for i in range(0, len(X), bs)])


class Adam:
    def __init__(self, net: Net, lr=1e-3, wd=0.0, b1=0.9, b2=0.999, eps=1e-8):
        self.net, self.lr, self.wd, self.b1, self.b2, self.eps = net, lr, wd, b1, b2, eps
        self.t = 0
        self.m, self.v = {}, {}

    def step(self):
        self.t += 1
        for layer, p, g in self.net.grads():
            w = getattr(layer, p)
            if self.wd:
                g = g + self.wd * w                     # torch.optim.Adam(weight_decay) — L2 jarima
            k = (id(layer), p)
            m = self.m.get(k, 0) * self.b1 + (1 - self.b1) * g
            v = self.v.get(k, 0) * self.b2 + (1 - self.b2) * g * g
            self.m[k], self.v[k] = m, v
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            setattr(layer, p, (w - self.lr * mh / (np.sqrt(vh) + self.eps)).astype(F32))


def dnn(d: int, k: int, seed: int = 42) -> Net:
    rng = np.random.default_rng(seed)
    return Net([Linear(d, 256, rng), BatchNorm(256), ReLU(), Dropout(0.3, rng),
                Linear(256, 128, rng), BatchNorm(128), ReLU(), Dropout(0.3, rng),
                Linear(128, 64, rng), ReLU(), Linear(64, k, rng)])


def autoencoder(d: int, seed: int = 42) -> Net:
    rng = np.random.default_rng(seed)
    return Net([Linear(d, 64, rng), ReLU(), Linear(64, 16, rng), ReLU(), Linear(16, 8, rng),
                Linear(8, 16, rng), ReLU(), Linear(16, 64, rng), ReLU(), Linear(64, d, rng)])


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_classifier(net: Net, Xa, ya, Xv, yv, k, epochs=30, lr=1e-3, bs=512, wts=None, wd=1e-5, patience=5, seed=42,
                   log=print):
    rng = np.random.default_rng(seed)
    opt = Adam(net, lr, wd)
    w = None if wts is None else np.asarray(wts, F32)
    best, bstate, wait, hist = 1e9, None, 0, []
    for ep in range(epochs):
        perm = rng.permutation(len(Xa))
        tl = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            x, y = Xa[idx], ya[idx]
            p = softmax(net.forward(x, train=True))
            n = len(y)
            if w is None:
                sw = np.full(n, 1.0 / n, F32)
            else:
                sw = w[y] / w[y].sum()
            loss = -(sw * np.log(p[np.arange(n), y] + 1e-12)).sum()
            g = p.copy()
            g[np.arange(n), y] -= 1
            net.backward((g * sw[:, None]).astype(F32))
            opt.step()
            tl += loss * n
        pv = softmax(net.predict(Xv))
        vl = float(-np.log(pv[np.arange(len(yv)), yv] + 1e-12).mean())
        hist.append((tl / len(Xa), vl))
        log(f"   davr {ep + 1:2d}: o‘quv {tl / len(Xa):.4f}, validatsiya {vl:.4f}")
        if vl < best - 1e-4:
            best, bstate, wait = vl, net.state(), 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.load(bstate)
    return net, hist


def fit_autoencoder(net: Net, Xn, Xvn, epochs=40, lr=1e-3, bs=512, patience=5, seed=42, log=print):
    rng = np.random.default_rng(seed)
    opt = Adam(net, lr)
    best, bstate, wait, hist = 1e9, None, 0, []
    for ep in range(epochs):
        perm = rng.permutation(len(Xn))
        tl = 0.0
        for i in range(0, len(perm), bs):
            x = Xn[perm[i:i + bs]]
            r = net.forward(x, train=True)
            d = r - x
            tl += float((d ** 2).mean()) * len(x)
            net.backward((2.0 * d / d.size).astype(F32))
            opt.step()
        vl = float(((net.predict(Xvn) - Xvn) ** 2).mean())
        hist.append((tl / len(Xn), vl))
        log(f"   davr {ep + 1:2d}: o‘quv {tl / len(Xn):.5f}, validatsiya {vl:.5f}")
        if vl < best - 1e-5:
            best, bstate, wait = vl, net.state(), 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.load(bstate)
    return net, hist


def recon_error(net: Net, X) -> np.ndarray:
    return ((net.predict(X) - X) ** 2).mean(1)
