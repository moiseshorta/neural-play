#!/usr/bin/env python3
"""
neuralplay - train the tiny timbre VAE used in lessons/11-autoencoder-vae.pd
Moisés Horta Valenzuela / SEMILLA.AI Studio

A Variational AutoEncoder (Kingma & Welling 2013) learns a 2-D map ("latent space")
of 8 organ-like timbres. A timbre here is just 8 numbers: the loudness of the first
8 harmonics. Encoder 8 -> 16 -> (mu, logvar) of a 2-D latent; decoder 2 -> 16 -> 8.

Run:  python3 train_vae.py          (needs numpy only, takes a few seconds)
It writes one text file per weight array into ../lessons/data/vae/, which the Pd lesson
loads into arrays and evaluates with [nn.dense]. Change the timbres below, retrain, reopen
the lesson - and Pd plays YOUR latent space.

Weight layout (what [nn.dense] expects): W has shape (n_out, n_in) and is saved row by
row, so W[j, i] is number j * n_in + i in the file.
"""
import os
import numpy as np

# numpy + Apple Accelerate can print bogus 'divide by zero in matmul' warnings on macOS; the
# results are checked below (gradient check + finiteness), so these warnings are silenced.
np.seterr(divide='ignore', over='ignore', invalid='ignore')

TIMBRES = {                      # 8 harmonic amplitudes each (0..1)
    'sine':     [1, 0, 0, 0, 0, 0, 0, 0],
    'saw':      [1, .5, .33, .25, .2, .17, .14, .12],
    'square':   [1, 0, .33, 0, .2, 0, .14, 0],
    'triangle': [1, 0, .11, 0, .04, 0, .02, 0],
    'organ':    [1, .8, .1, .6, 0, .3, 0, .5],
    'hollow':   [.2, 1, .7, .1, 0, 0, 0, 0],
    'bright':   [.3, .4, .5, .6, .7, .8, .9, 1],
    'odd':      [1, 0, 0, .8, 0, 0, .6, .3],
}
H, Z = 16, 2                     # hidden width, latent size
BETA = 0.05                      # weight of the KL term (0 = plain autoencoder)
STEPS, LR, BATCH = 6000, 0.01, 256
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lessons', 'data', 'vae')


def data(rng, n):
    """augmented training batch: each timbre slightly varied, so the VAE learns smooth regions"""
    base = np.array(list(TIMBRES.values()))
    x = base[rng.integers(len(base), size=n)]
    return np.clip(x * (1 + 0.1 * rng.standard_normal(x.shape)) + 0.02 * rng.standard_normal(x.shape), 0, 1)


def init(rng):
    def W(o, i): return rng.standard_normal((o, i)) * np.sqrt(1 / i)
    return dict(W1=W(H, 8), b1=np.zeros(H), Wm=W(Z, H), bm=np.zeros(Z), Wl=W(Z, H), bl=np.zeros(Z),
                W2=W(H, Z), b2=np.zeros(H), W3=W(8, H), b3=np.zeros(8))


def forward(P, x, eps):
    h = np.tanh(x @ P['W1'].T + P['b1'])
    mu, lv = h @ P['Wm'].T + P['bm'], h @ P['Wl'].T + P['bl']
    z = mu + np.exp(0.5 * lv) * eps                       # the reparameterisation trick
    d = np.tanh(z @ P['W2'].T + P['b2'])
    y = 1 / (1 + np.exp(-(d @ P['W3'].T + P['b3'])))
    return h, mu, lv, z, d, y


def loss_and_grads(P, x, eps):
    n = len(x)
    h, mu, lv, z, d, y = forward(P, x, eps)
    recon = ((y - x) ** 2).sum(1)
    kl = -0.5 * (1 + lv - mu ** 2 - np.exp(lv)).sum(1)
    L = (recon + BETA * kl).mean()
    g = {}
    da3 = 2 * (y - x) * y * (1 - y) / n                    # back through the sigmoid
    g['W3'], g['b3'] = da3.T @ d, da3.sum(0)
    da2 = (da3 @ P['W3']) * (1 - d ** 2)                   # back through tanh
    g['W2'], g['b2'] = da2.T @ z, da2.sum(0)
    dz = da2 @ P['W2']
    dmu = dz + BETA * mu / n
    dlv = dz * eps * 0.5 * np.exp(0.5 * lv) + BETA * 0.5 * (np.exp(lv) - 1) / n
    g['Wm'], g['bm'] = dmu.T @ h, dmu.sum(0)
    g['Wl'], g['bl'] = dlv.T @ h, dlv.sum(0)
    da1 = (dmu @ P['Wm'] + dlv @ P['Wl']) * (1 - h ** 2)
    g['W1'], g['b1'] = da1.T @ x, da1.sum(0)
    return L, recon.mean(), kl.mean(), g


def gradient_check(rng):
    """compare the hand-written gradients with finite differences (they must agree)"""
    P = init(rng); x = data(rng, 5); eps = rng.standard_normal((5, Z))
    _, _, _, g = loss_and_grads(P, x, eps)
    for k in P:
        idx = tuple(rng.integers(s) for s in P[k].shape)
        old = P[k][idx]; P[k][idx] = old + 1e-5; Lp = loss_and_grads(P, x, eps)[0]
        P[k][idx] = old - 1e-5; Lm = loss_and_grads(P, x, eps)[0]; P[k][idx] = old
        num = (Lp - Lm) / 2e-5
        assert abs(num - g[k][idx]) < 1e-6 + 1e-3 * abs(num), (k, num, g[k][idx])


def train(seed=1):
    rng = np.random.default_rng(seed)
    gradient_check(rng)
    P = init(rng)
    m = {k: np.zeros_like(v) for k, v in P.items()}; v = {k: np.zeros_like(p) for k, p in P.items()}
    for t in range(1, STEPS + 1):                           # Adam (Kingma & Ba 2014)
        x = data(rng, BATCH)
        L, rec, kl, g = loss_and_grads(P, x, rng.standard_normal((BATCH, Z)))
        for k in P:
            m[k] = 0.9 * m[k] + 0.1 * g[k]; v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
            P[k] -= LR * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
        if t % 1000 == 0:
            print(f'step {t}: loss {L:.4f}  reconstruction {rec:.4f}  KL {kl:.3f}')
    return P


def save(P):
    os.makedirs(OUT, exist_ok=True)
    names = {'W1': 'eh_w', 'b1': 'eh_b', 'Wm': 'mu_w', 'bm': 'mu_b', 'Wl': 'lv_w', 'bl': 'lv_b',
             'W2': 'dh_w', 'b2': 'dh_b', 'W3': 'y_w', 'b3': 'y_b'}
    for k, n in names.items():
        np.savetxt(os.path.join(OUT, n + '.txt'), P[k].reshape(1, -1), fmt='%.6f')


if __name__ == '__main__':
    P = train()
    assert all(np.all(np.isfinite(p)) for p in P.values()), 'training diverged'
    save(P)
    base = np.array(list(TIMBRES.values()))
    _, mu, lv, _, _, _ = forward(P, base, np.zeros((len(base), Z)))
    _, _, _, _, _, rec = forward(P, base, np.zeros((len(base), Z)))
    for name, m_, s_, r in zip(TIMBRES, mu, np.exp(0.5 * lv), rec):
        print(f'{name:9s} latent ({m_[0]:+.2f}, {m_[1]:+.2f})  spread {s_.mean():.2f}  error {np.abs(r - TIMBRES[name]).max():.2f}')
    print('weights written to', os.path.normpath(OUT))
