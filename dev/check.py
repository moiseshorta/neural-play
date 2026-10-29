#!/usr/bin/env python3
"""Headless checks for the neuralplay Pd package.

Runs Pd in -batch mode (faster than real time, no audio device), renders the
lessons to WAV through a shadow [output~] (dev/shadow/output~.pd) and checks the
numbers / audio. Usage:  python3 dev/check.py  [name-filter]
Needs: Pd 0.54+ (PD env var or the default macOS app path), numpy.
"""
import math, os, re, struct, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), 'neuralplay')
PD = os.environ.get('PD', '/Applications/Pd-0.55-2.app/Contents/Resources/bin/pd')
SR = 48000


def run_pd(patches, timeout=120, extra=()):
    cmd = [PD, '-batch', '-nogui', '-noaudio', '-stderr', '-r', str(SR),
           '-path', os.path.join(HERE, 'shadow'), *extra]
    for p in patches:
        cmd += ['-open', p]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.stderr


def read_wav(path):
    b = open(path, 'rb').read(); i = 12; ch = bits = None; d = b''
    while i < len(b) - 8:
        cid, n = b[i:i + 4], struct.unpack('<I', b[i + 4:i + 8])[0]
        if cid == b'fmt ': _, ch, _, _, _, bits = struct.unpack('<HHIIHH', b[i + 8:i + 24])
        if cid == b'data': d = b[i + 8:i + 8 + n]; break
        i += 8 + n + (n & 1)
    return np.frombuffer(d, '<f4').reshape(-1, ch).astype(np.float64)


def render(lesson, seconds, events=(), timeout=180, counts=()):
    """Open a lesson plus a generated recorder patch; events = [(ms, 'receiver message ...')].
    counts = receive names: every non-zero message to them is counted, printed at the end as 'COUNT-name: n'."""
    tmp = tempfile.mkdtemp()
    wav = os.path.join(tmp, 'out.wav')
    n_samp = int(seconds * SR)
    # record into arrays with [tabwrite~] (sample-exact), then write them with [soundfiler],
    # which is synchronous - [writesf~] writes on a thread and can lose data when batch mode quits
    lines = ['#N canvas 0 50 400 300 12;',
             '#X obj 10 10 catch~ nntest-L;', '#X obj 120 10 catch~ nntest-R;',
             '#X obj 10 40 tabwrite~ nntest-bufL;', '#X obj 200 60 loadbang;',
             '#X msg 200 90 \\; pd dsp 1;', '#X obj 120 40 tabwrite~ nntest-bufR;',
             f'#X obj 10 70 array define nntest-bufL {n_samp};', f'#X obj 10 90 array define nntest-bufR {n_samp};',
             f'#X msg 10 110 write -bytes 4 {wav} nntest-bufL nntest-bufR;',
             '#X obj 10 140 soundfiler;', '#X msg 10 170 \\; pd quit;']
    conns = ['#X connect 0 0 2 0;', '#X connect 1 0 5 0;', '#X connect 3 0 4 0;',
             '#X connect 3 0 2 0;', '#X connect 3 0 5 0;', '#X connect 8 0 9 0;']
    n = 11
    for ms, m in list(events):
        lines.append(f'#X obj 200 {120 + n * 2} delay {ms};')
        lines.append(f'#X msg 200 {140 + n * 2} \\; ' + m.replace(',', ' \\,').replace(';', ' \\;') + ';')
        conns += [f'#X connect 3 0 {n} 0;', f'#X connect {n} 0 {n + 1} 0;']
        n += 2
    # at the end: write the file (blocking), then quit
    lines += [f'#X obj 300 10 delay {seconds * 1000 + 10};', '#X obj 300 40 t b b;']
    conns += [f'#X connect 3 0 {n} 0;', f'#X connect {n} 0 {n + 1} 0;', f'#X connect {n + 1} 1 8 0;',
              f'#X connect {n + 1} 0 10 0;']
    n += 2
    for name in counts:   # [r name] -> [!= 0] -> running sum [+ ] -> [f] ; printed by the final delay
        lines += [f'#X obj 10 {200 + n} r {name};', '#X obj 10 10 != 0;', '#X obj 10 10 + 0;', '#X obj 10 10 t f f;',
                  '#X obj 10 10 f;', f'#X obj 10 10 print COUNT-{name};', f'#X obj 10 10 delay {seconds * 1000 - 1};']
        conns += [f'#X connect {n} 0 {n+1} 0;', f'#X connect {n+1} 0 {n+2} 0;', f'#X connect {n+2} 0 {n+3} 0;',
                  f'#X connect {n+3} 1 {n+2} 1;', f'#X connect {n+3} 0 {n+4} 1;', f'#X connect {n+4} 0 {n+5} 0;',
                  f'#X connect 3 0 {n+6} 0;', f'#X connect {n+6} 0 {n+4} 0;']
        n += 7
    rec = os.path.join(tmp, 'rec.pd')
    open(rec, 'w').write('\n'.join(lines + conns) + '\n')
    log = run_pd([os.path.join(PKG, lesson), rec], timeout=timeout)
    return read_wav(wav), log


def stats(x):
    m = x.mean(axis=1)
    return dict(rms=float(np.sqrt(np.mean(m ** 2))), peak=float(np.max(np.abs(m))),
                dc=float(np.mean(m)), finite=bool(np.all(np.isfinite(m))))


def spectrum_peak(x, lo=20, hi=5000):
    m = x.mean(axis=1); m = m - m.mean()
    w = np.abs(np.fft.rfft(m * np.hanning(len(m))))
    f = np.fft.rfftfreq(len(m), 1 / SR)
    band = (f > lo) & (f < hi)
    return float(f[band][np.argmax(w[band])])


def errors_in(log):
    bad = [l for l in log.splitlines()
           if re.search(r"couldn't create|error|bad arguments|no such|DSP loop|not found|connection failed|multiply defined", l, re.I)
           and 'nn.act~-unknown-function' not in l]
    return bad


# ----------------------------------------------------------------------------- checks
ACT = {
    'linear': lambda x: x, 'step': lambda x: float(x > 0), 'sign': lambda x: 1.0 if x > 0 else -1.0,
    'sigmoid': lambda x: 1 / (1 + math.exp(-x)), 'tanh': math.tanh, 'relu': lambda x: max(x, 0),
    'leaky': lambda x: max(x, 0.1 * x), 'softsign': lambda x: x / (1 + abs(x)), 'sin': math.sin,
    'gauss': lambda x: math.exp(-x * x), 'elu': lambda x: x if x > 0 else math.exp(x) - 1,
    'gelu': lambda x: 0.5 * x * (1 + math.tanh(0.7978846 * (x + 0.044715 * x ** 3))),
    'swish': lambda x: x / (1 + math.exp(-x)), 'hardclip': lambda x: min(max(x, -1), 1),
}


def check_act():
    log = run_pd([os.path.join(HERE, 'tests', 'test_act.pd')])
    assert not errors_in(log), errors_in(log)
    x = None; n = 0
    for line in log.splitlines():
        if line.startswith('STEP:'): x = float(line.split()[1])
        elif line.startswith('T:'):
            name, y = line.split()[-2], float(line.split()[-1])
            want = ACT[name](max(-8, min(8, x)))
            assert abs(y - want) < 2e-3, f'{name}({x}) = {y}, want {want}'
            n += 1
    assert n == 14 * 9, f'only {n} values checked'
    return f'{n} activation values match'


def printed(log, tag):
    return [float(l.split()[-1]) for l in log.splitlines() if l.startswith(tag + ':')]


def check_neuron():
    log = run_pd([os.path.join(HERE, 'tests', 'test_neuron.pd')])
    assert not errors_in(log), errors_in(log)
    a, b = printed(log, 'A')[0], printed(log, 'B')[0]
    assert abs(a - -0.65) < 1e-4, f'weighted sum {a}'                 # 0.5 - 2 + 0.75 + 1 - 1 + 0.1
    assert abs(b - math.tanh(-0.65)) < 1e-3, f'tanh neuron {b}'
    assert printed(log, 'X') == [0, 1, 1, 0], f"XOR net gave {printed(log, 'X')}"
    e10, e50 = printed(log, 'E')
    assert abs(e10 - (1 - math.exp(-1))) < 0.06 and abs(e50 - (1 - math.exp(-5))) < 0.01, f'leak {e10} {e50}'
    w = [l for l in log.splitlines() if l.startswith('nn.neuron~: w')]
    assert len(w) == 3 and w[1] == w[2] and w[0] != w[1], f'seeding: {w}'
    return 'sum, tanh, XOR 0110, tau step response, seeded randomize all ok'


def check_izh():
    log = run_pd([os.path.join(HERE, 'tests', 'test_izh.pd')])
    assert not errors_in(log), errors_in(log)
    # spike counts in 2 s at I = 10, from an independent Python implementation of the same model
    want = {'NRS1000': 45, 'NRS10000': 439, 'NFS1000': 269, 'NCH1000': 172}
    got = {k: printed(log, k)[0] for k in want}
    assert all(abs(got[k] - want[k]) <= 1 for k in want), f'spike counts {got} want {want}'
    return f'spike counts match reference model {got}'


def onsets(x, thr=0.05, frame=240, refractory=10):
    m = np.abs(x.mean(axis=1)); n = len(m) // frame
    env = m[:n * frame].reshape(n, frame).max(axis=1)
    count, last = 0, -refractory
    for i in range(1, n):
        if env[i] > thr and env[i] > 1.4 * env[i - 1] + 0.01 and i - last >= refractory:
            count += 1; last = i
    return count


def check_l01():
    ev = [(50, 'l01-inputs 0'), (100, 'l01-play 1')]
    x, log = render('lessons/01-mcculloch-pitts.pd', 6.0, ev, counts=['l01-fire'])
    assert not errors_in(log), errors_in(log)
    n_and, heard = printed(log, 'COUNT-l01-fire')[0], onsets(x)
    x, log = render('lessons/01-mcculloch-pitts.pd', 6.0, ev + [(60, 'l01-weights 1 1 0 1')], counts=['l01-fire'])
    n_or = printed(log, 'COUNT-l01-fire')[0]
    # 48 sixteenths in 6 s: AND = steps divisible by 12 (4), OR = by 4 or 3 (12 + 16 - 4 = 24)
    assert n_and == 4 and n_or == 24, f'AND fired {n_and} (want 4), OR fired {n_or} (want 24)'
    assert heard == 4 and stats(x)['finite'], f'heard {heard} bells for AND'
    return f'AND fires {n_and:g}x (4 bells heard), OR fires {n_or:g}x in 6 s'



def check_l02():
    lp = 'lessons/02-activation-functions.pd'
    x, log = render(lp, 2.0)
    assert not errors_in(log), errors_in(log)
    f_tanh = spectrum_peak(x[SR // 2:])
    g, _ = render(lp, 2.0, [(100, 'l02-act gauss')])
    f_gauss = spectrum_peak(g[SR // 2:])
    d, _ = render(lp, 2.0, [(100, 'l02-listen 1')])
    a = x[SR // 2:, 0]   # the drawing path is read through [r~]: one block (64 samples) later
    r = max(np.corrcoef(a[:-128], d[SR // 2 + lag:len(d) - 128 + lag, 0])[0, 1] for lag in range(0, 128))
    assert abs(f_tanh - 110) < 3 and abs(f_gauss - 220) < 4, f'peaks tanh {f_tanh} gauss {f_gauss}'
    assert r > 0.99 and stats(x)['rms'] > 0.01, f'drawn curve vs nn.act~ corr {r}'
    return f'tanh peak {f_tanh:.0f} Hz, gauss (even curve) {f_gauss:.0f} Hz = octave up, drawn tanh ~ nn.act~ (r={r:.3f})'



def check_l03():
    lp = 'lessons/03-the-neuron.pd'
    a, log = render(lp, 2.0, [(5, 'l03-listen 0')])
    assert not errors_in(log), errors_in(log)
    b, _ = render(lp, 2.0, [(5, 'l03-listen 1')])
    d = np.max(np.abs(a[SR // 2:] - b[SR // 2:]))
    assert d < 1e-3 and stats(a)['rms'] > 0.01, f'hand-made vs nn.neuron~ differ by {d}'
    return f'hand-made neuron == [nn.neuron~] (max difference {d:.1e})'



def check_ctl():
    log = run_pd([os.path.join(HERE, 'tests', 'test_ctl.pd')])
    assert not errors_in(log), errors_in(log)
    got = lambda tag: [list(map(float, l.split()[1:])) for l in log.splitlines() if l.startswith(tag + ':')]
    act = [v[0] for v in got('ACT')]
    want = [0, 3, 0.5, ACT['gelu'](1)]
    assert np.allclose(act, want, atol=1e-4), f'nn.act {act}'
    assert np.allclose(got('DENSE')[0], [4.6, -0.7], atol=1e-5), got('DENSE')
    assert np.allclose(got('DENSETANH')[0], np.tanh([4.6, -0.7]), atol=1e-5)
    sc = np.array([1, 2, 3, 0.0])
    for T, p in zip([1, 0.1], got('SOFTMAX')):
        e = np.exp((sc - sc.max()) / T); assert np.allclose(p, e / e.sum(), atol=1e-5), (T, p)
    return 'nn.act values, nn.dense matrix product (linear + tanh), nn.softmax at T=1 and T=0.1 all match numpy'



def check_l04():
    got = {}
    for gate in ['AND', 'OR', 'NAND', 'XOR']:
        x, log = render('lessons/04-perceptron.pd', 8.0, [(5, f'l04-gate {gate}'), (10, 'l04-speed 10'), (20, 'l04-train 1')])
        assert not errors_in(log) and stats(x)['finite'] and stats(x)['rms'] > 0.01, (gate, errors_in(log))
        ep = printed(log, 'perceptron-solved-at-epoch')
        got[gate] = int(ep[0]) if ep else None
    # independent Python perceptron with the same start (0.2, -0.4, 0.1) and rate (slider: 0.248) gives 8, 5, 4, never
    assert got == {'AND': 8, 'OR': 5, 'NAND': 4, 'XOR': None}, got
    return f'solved at epoch AND 8, OR 5, NAND 4 (= Python reference), XOR never'



def band(x, hz, width=5):
    m = x.mean(axis=1); m = m - m.mean()
    w = np.abs(np.fft.rfft(m * np.hanning(len(m)))); f = np.fft.rfftfreq(len(m), 1 / SR)
    return float(w[(f > hz - width) & (f < hz + width)].max())


def check_l05():
    lp = 'lessons/05-hidden-layer-xor.pd'
    x, log = render(lp, 3.0)
    assert not errors_in(log) and stats(x)['rms'] > 0.05, errors_in(log)
    r, _ = render(lp, 2.0, [(5, 'l05-inlevel 0'), (5, 'l05-xorlevel 0'), (5, 'l05-raw 1'), (5, 'l05-speed 100')])
    r = r[SR:]
    diff, summ, inp = band(r, 100), band(r, 500), band(r, 200)
    # XOR of bipolar squares = their product: sum and difference tones, but not the input frequency
    assert diff > 20 * inp and summ > 20 * inp, (diff, summ, inp)
    return f'XOR of 200 Hz and 300 Hz pulses = ring modulation: 100 Hz and 500 Hz present, 200 Hz absent ({diff / inp:.0f}x)'



def backprop_reference(start, lr, target=(0, 1, 1, 0), epochs=5000):
    """plain-Python 2-3-1 backprop, same order and formulas as lesson 06; returns epoch where loss < 0.02"""
    names = [f'{n}{k}' for k in (1, 2, 3) for n in ('wa', 'wb', 'hb', 'v')] + ['c']
    P = dict(zip(names, start))
    for ep in range(1, epochs):
        L = 0
        for p in range(4):
            x1, x2, t = p // 2, p % 2, target[p]
            h = [math.tanh(P[f'wa{k}'] * x1 + P[f'wb{k}'] * x2 + P[f'hb{k}']) for k in (1, 2, 3)]
            y = 1 / (1 + math.exp(-(sum(P[f'v{k}'] * h[k - 1] for k in (1, 2, 3)) + P['c'])))
            dy = (y - t) * y * (1 - y); L += (y - t) ** 2 / 2
            dh = [dy * P[f'v{k}'] * (1 - h[k - 1] ** 2) for k in (1, 2, 3)]
            for k in (1, 2, 3): P[f'v{k}'] -= lr * dy * h[k - 1]
            P['c'] -= lr * dy
            for k in (1, 2, 3):
                P[f'wa{k}'] -= lr * dh[k - 1] * x1; P[f'wb{k}'] -= lr * dh[k - 1] * x2; P[f'hb{k}'] -= lr * dh[k - 1]
        if L < 0.02: return ep
    return None


def check_l06():
    x, log = render('lessons/06-backprop.pd', 12.0, [(5, 'l06-steps 100'), (10, 'l06-train 1'), (11990, 'l06-print bang')])
    assert not errors_in(log) and stats(x)['finite'], errors_in(log)
    line = lambda tag: [list(map(float, l.split()[1:])) for l in log.splitlines() if l.startswith(tag + ':')]
    start, solved, answers = line('backprop-start')[0], line('backprop-solved-at-epoch')[0][0], line('backprop-answers')[-1]
    lr = 0.05 * math.exp(round(math.log(10) / math.log(100) * 119) / 119 * math.log(100))   # slider start 0.5, pixel-rounded
    ref = backprop_reference(start, lr)
    assert ref is not None and abs(ref - solved) <= 2, f'Pd solved at {solved}, Python reference at {ref}'
    assert answers[0] < 0.1 and answers[1] > 0.9 and answers[2] > 0.9 and answers[3] < 0.1, answers
    return f'Pd learns XOR at epoch {solved:g} = Python backprop {ref}; answers {[round(a, 3) for a in answers]}'



def check_l07():
    lp = 'lessons/07-hopfield-memory.pd'
    for k, cue in enumerate('ABC', 1):
        ev = [(10, f'l07-cue {cue}'), (20, 'l07-corrupt bang')]
        x, log = render(lp, 3.0, ev)
        assert not errors_in(log) and not printed(log, 'hopfield-recalled'), ('corrupted state should not be a memory', cue)
        x, log = render(lp, 8.0, ev + [(30, 'l07-rate 100'), (40, 'l07-recall 1')])
        assert not errors_in(log) and printed(log, 'hopfield-recalled') == [k], (cue, printed(log, 'hopfield-recalled'))
        assert stats(x)['rms'] > 0.02
    return 'grooves A, B, C each recalled exactly after 6 of 32 steps were flipped'



def centroid(seg):
    w = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))); f = np.fft.rfftfreq(len(seg), 1 / SR)
    return float((w * f).sum() / w.sum())


def check_l08():
    lp = 'lessons/08-mlp-mapping.pd'
    x, log = render(lp, 4.0, [(10, 'l08-explore 1')])
    assert not errors_in(log) and stats(x)['finite'] and stats(x)['rms'] > 0.005, errors_in(log)
    m = x.mean(axis=1)
    cs = [centroid(m[i:i + 4800]) for i in range(4800, len(m) - 4800, 4800)]
    a, _ = render(lp, 1.5, [(10, 'l08-x -1'), (10, 'l08-y -1')])
    b, _ = render(lp, 1.5, [(10, 'l08-x 1'), (10, 'l08-y 1')])
    c2, _ = render(lp, 1.5, [(5, 'nn-l08 seed 7, randomize'), (10, 'l08-x -1'), (10, 'l08-y -1')])
    assert max(cs) - min(cs) > 50, ('explore does not change the sound', min(cs), max(cs))
    assert np.max(np.abs(a - b)) > 0.01 and np.max(np.abs(a - c2)) > 0.01, 'XY or seed has no audible effect'
    return f'explore sweeps the brightness {min(cs):.0f}-{max(cs):.0f} Hz; XY corners and seeds give different sounds'



def f0_autocorr(x, lo=20, hi=2000):
    seg = x.mean(axis=1); seg = seg - seg.mean(); seg = seg[:SR // 2]
    ac = np.correlate(seg, seg, 'full')[len(seg) - 1:]
    z = int(np.argmax(ac < 0))                  # first zero crossing: skip the trivial peak at lag 0
    b = SR // lo
    return SR / (np.argmax(ac[z:b]) + z)


def check_l09():
    lp = 'lessons/09-recurrent-ctrnn.pd'
    got = {}
    for tau, want in [(1, 35), (0.3, 114), (0.1, 342)]:   # frequencies from a Python simulation of the same CTRNN
        x, log = render(lp, 2.5, [(5, f'l09-tau {tau}'), (5, 'l09-voice 0'), (5, 'l09-direct 1')])
        assert not errors_in(log), errors_in(log)
        got[tau] = f0_autocorr(x[SR:])
        assert abs(got[tau] - want) / want < 0.08, (tau, got[tau], want)
    q, _ = render(lp, 2.5, [(5, 'l09-tau 0.1'), (5, 'l09-voice 0'), (5, 'l09-direct 1'), (10, 'nn-l09a w 1 0.5 1'), (10, 'nn-l09b w -0.5 1 0')])
    assert stats(q[SR:])['rms'] < 0.005, ('weak weights should settle', stats(q[SR:]))
    return 'oscillates at ' + ', '.join(f'{v:.0f} Hz (tau {k} ms)' for k, v in got.items()) + ' as simulated; weak weights settle'



def rate_from_envelope(x, skip_ms=1000, max_lag_ms=200):
    m = np.abs(x.mean(axis=1)); fr = SR // 1000
    env = m[:len(m) // fr * fr].reshape(-1, fr).max(axis=1)[skip_ms:]
    env = env - env.mean(); ac = np.correlate(env, env, 'full')[len(env) - 1:]
    z = int(np.argmax(ac < 0)); return 1000 / (np.argmax(ac[z:max_lag_ms]) + z)


def check_l10():
    lp = 'lessons/10-spiking-neurons.pd'
    out = []
    for I, tau in [(1.5, 20), (2.0, 10)]:
        theory = 1000 / (tau * math.log(I / (I - 1)))
        x, log = render(lp, 3.0, [(5, 'l10-izhlevel 0'), (5, f'l10-I {I}'), (5, f'l10-tau {tau}')])
        assert not errors_in(log), errors_in(log)
        got = rate_from_envelope(x)
        assert abs(got - theory) / theory < 0.05, (I, tau, got, theory)
        out.append(f'{got:.1f} Hz (theory {theory:.1f})')
    y, _ = render(lp, 3.0, [(5, 'l10-liflevel 0')])
    assert stats(y)['rms'] > 0.02 and stats(y)['finite'], 'Izhikevich pings silent'
    return 'LIF fires at ' + ', '.join(out) + '; Izhikevich pings audible'



def vae_reference(spectrum):
    """numpy forward pass with the weight files the lesson loads: returns (mu, decoded spectrum at mu)"""
    d = os.path.join(PKG, 'lessons', 'data', 'vae')
    W = {n: np.loadtxt(os.path.join(d, n + '.txt')) for n in ['eh_w', 'eh_b', 'mu_w', 'mu_b', 'dh_w', 'dh_b', 'y_w', 'y_b']}
    h = np.tanh(W['eh_w'].reshape(16, 8) @ spectrum + W['eh_b'])
    mu = W['mu_w'].reshape(2, 16) @ h + W['mu_b']
    dh = np.tanh(W['dh_w'].reshape(16, 2) @ mu + W['dh_b'])
    return mu, 1 / (1 + np.exp(-(W['y_w'].reshape(8, 16) @ dh + W['y_b'])))


def check_l11():
    lp = 'lessons/11-autoencoder-vae.pd'
    x, log = render(lp, 3.0, [(10, 'l11-preset organ'), (2900, 'l11-print bang')])
    assert not errors_in(log) and stats(x)['rms'] > 0.01, errors_in(log)
    line = lambda tag: [list(map(float, l.split()[1:])) for l in log.splitlines() if l.startswith(tag + ':')][-1]
    mu_pd, y_pd = np.array(line('vae-mu')), np.array(line('vae-y'))
    mu, y = vae_reference(np.array([1, 0.8, 0.1, 0.6, 0, 0.3, 0, 0.5]))
    assert np.allclose(mu_pd, mu, atol=1e-3) and np.allclose(y_pd, y, atol=2e-3), (mu_pd, mu, y_pd, y)
    return f'Pd encoder/decoder = numpy with the same weights (organ -> mu {np.round(mu_pd, 3)}, decoded within 2e-3)'



def check_l12():
    got = []
    for mean in (64, 52):
        x, log = render('lessons/12-gan.pd', 12.0, [(5, f'l12-mean {mean}'), (5, 'l12-steps 100'), (10, 'l12-train 1'), (11900, 'l12-print bang')])
        assert not errors_in(log) and stats(x)['rms'] > 0.01, errors_in(log)
        m, sd = [list(map(float, l.split()[1:])) for l in log.splitlines() if l.startswith('gan-forger:')][-1]
        assert abs(m - mean) < 2.5 and 0.05 < sd < 12, (mean, m, sd)
        got.append(f'teacher {mean} -> forger {m:.1f}')
    return 'forger starts at MIDI 40 and learns the teacher: ' + ', '.join(got)



def check_l13():
    lp = 'lessons/13-attention.pd'
    peaks = {}
    for (qx, qy), want in [((1, 1), 110), ((-1, 1), 220), ((1, -1), 440), ((-1, -1), 1000)]:
        x, log = render(lp, 2.0, [(5, f'l13-qx {qx}'), (5, f'l13-qy {qy}'), (5, 'l13-temp 0.05')])
        assert not errors_in(log), errors_in(log)
        peaks[want] = spectrum_peak(x[SR:], 50, 3000)
        assert abs(peaks[want] - want) / want < 0.05, (qx, qy, peaks[want])
    b, _ = render(lp, 2.0, [(5, 'l13-qx 1'), (5, 'l13-qy 1'), (5, 'l13-temp 5')])
    lvl = [band(b[SR:], f) for f in (110, 220, 440)]
    assert min(lvl) > 0.1 * max(lvl), ('high temperature should blend all sounds', lvl)
    return 'T = 0.05 selects one value per corner (peaks ' + ', '.join(f'{v:.0f}' for v in peaks.values()) + ' Hz), T = 5 blends them'



def induction_reference(prompt, n, ctx=2.0, rec=0.05):
    """the same induction-head rule at temperature -> 0 (argmax), in plain Python"""
    seq = list(prompt)
    for _ in range(n):
        L = len(seq); c, p = seq[-1], seq[-2]
        score = lambda j: -4 * abs(seq[j - 1] - c) / 12 - ctx * (j >= 2) * abs(seq[j - 2] - p) / 12 - rec * (L - 1 - j)
        seq.append(seq[max(range(1, L), key=score)])
    return seq[len(prompt):]


def check_l14():
    x, log = render('lessons/14-transformer-induction.pd', 3.6, [(5, 'l14-temp 0.002'), (10, 'l14-gen 1'), (3550, 'l14-print bang')])
    assert not errors_in(log) and stats(x)['rms'] > 0.01, errors_in(log)
    seq = [int(float(v)) for v in [l for l in log.splitlines() if l.startswith('induction-seq:')][-1].split()[1:]]
    L = int(float([l for l in log.splitlines() if l.startswith('induction-L:')][-1].split()[1]))
    prompt = [60, 62, 64, 60, 60, 62, 64, 60, 64, 65, 67, 64, 65, 67]
    assert seq[:14] == prompt and L > 20, (seq, L)
    want = induction_reference(prompt, L - 14)
    assert seq[14:L] == want, (seq[14:L], want)
    return f'continues the prompt with {seq[14:L][:9]}... = Python reference ({L - 14} notes)'



def check_l15():
    chords = {'Cmaj': [48, 52, 55, 60, 64, 67, 72, 76, 79], 'whole': [48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70]}
    out = []
    for name, tones in chords.items():
        x, log = render('lessons/15-diffusion.pd', 7.5, [(20, f'l15-chord {name}'), (50, 'l15-go bang')])
        assert not errors_in(log) and stats(x)['rms'] > 0.02, errors_in(log)[:3]
        final = [float(v) for v in [l for l in log.splitlines() if l.startswith('diffusion-final:')][-1].split()[1:]]
        off = max(min(abs(f - t) for t in tones) for f in final)
        assert len(final) == 6 and off < 0.1, (name, final, off)
        out.append(f'{name} -> {[round(f, 1) for f in final]}')
    return 'random clouds denoised onto chord tones: ' + '; '.join(out)



def check_l16():
    lp = 'lessons/16-feedback-reservoir.pd'
    x, log = render(lp, 6.0)
    assert not errors_in(log) and stats(x)['finite'] and 0.01 < stats(x)['rms'] and stats(x)['peak'] < 1, (errors_in(log), stats(x))
    y, _ = render(lp, 6.0, [(5, 'l16-fb 0')])
    p, _ = render(lp, 6.0, [(5, 'l16-mixA 0'), (5, 'l16-mixB 0'), (5, 'l16-mixC 1')])
    assert np.max(np.abs(x - y)) > 0.05, 'feedback amount has no effect'
    assert stats(p)['rms'] > 0.005, 'bob~ pings are silent'
    return f'27-neuron feedback synth stable (rms {stats(x)["rms"]:.3f}), feedback changes the sound, bob~ pings audible (rms {stats(p)["rms"]:.3f})'



# ----------------------------------------------------------------------------- lesson 17: Q-learning
L17 = 'lessons/17-reinforcement-learning.pd'
L17_MOVES = np.array([-2, -1, 1, 2])


def q_optimal(goal, gamma):
    """value iteration on the same world: the exact values Q-learning should converge to"""
    Q = np.zeros((8, 4))
    for _ in range(100):
        V = Q.max(1); V[goal] = 0
        s2 = np.clip(np.arange(8)[:, None] + L17_MOVES, 0, 7)
        Q = (s2 == goal) + gamma * (s2 != goal) * V[s2]
    return Q


def q_replay(moves, alpha, gamma, eps, goal):
    """replays every traced move with numpy: the world, the choice and the update. Returns (worst error, explore rate)"""
    Q = np.zeros((8, 4)); worst = 0.0; nex = 0
    for s, a, ex, r, s2, dn, qn, dl in moves:
        s, a, s2 = int(s), int(a), int(s2)
        assert s != goal and s2 == min(max(s + L17_MOVES[a], 0), 7), ('world', s, a, s2)
        assert r == float(s2 == goal) and dn == r, ('reward', s, a, s2, r, dn)
        assert ex or Q[s, a] >= Q[s].max() - 2e-5, ('greedy move was not a best move', s, a, Q[s])
        want_dl = r + (1 - dn) * gamma * Q[s2].max() - Q[s, a]
        want_q = Q[s, a] + alpha * want_dl
        worst = max(worst, abs(want_dl - dl), abs(want_q - qn))
        Q[s, a] = qn; nex += ex           # continue from Pd's value (printed with 6 digits): every step is checked on its own
    return worst, nex / len(moves), Q


def l17_run(events, seconds=1.0):
    x, log = render(L17, seconds, events)
    assert not errors_in(log), errors_in(log)[:3]
    moves = [list(map(float, l.split()[1:])) for l in log.splitlines() if l.startswith('l17-move:')]
    qs = [np.array(list(map(float, l.split()[1:]))).reshape(8, 4) for l in log.splitlines() if l.startswith('l17-q:')]
    return x, log, moves, qs


def check_l17():
    # 1) 400 practice phrases traced move by move: the world, the epsilon-greedy choice and every update = numpy
    al, ga, eps, goal = 0.5, 0.9, 0.5, 2
    x, log, moves, qs = l17_run([(10, f'l17-alpha {al}'), (10, f'l17-gamma {ga}'), (10, f'l17-eps {eps}'), (10, f'l17-goal {goal}'),
                                 (20, 'l17-trace 1'), (30, 'l17-practice 400'), (40, 'l17-dump bang'), (50, 'l17-trace 0'),
                                 (60, 'l17-good bang'), (70, 'l17-dump bang')])
    worst, rate, Q = q_replay(moves, al, ga, eps, goal)
    n = len(moves); sd = np.sqrt(eps * (1 - eps) / n)
    assert n > 1000 and worst < 2e-5 and abs(rate - eps) < 4 * sd, (n, worst, rate)
    assert np.abs(qs[0] - Q).max() < 1e-6, 'dumped table != traced table'
    Qs = q_optimal(goal, ga); V, Vs = qs[0].max(1), Qs.max(1); V[goal] = Vs[goal] = 0
    assert np.abs(V - Vs).max() < 0.01, ('values did not converge to the optimum', V, Vs)
    # 2) good! adds alpha to the last move only
    hum = [list(map(float, l.split()[1:])) for l in log.splitlines() if l.startswith('l17-human:')][-1]
    d = qs[1] - qs[0]; ls, la = int(hum[0]), int(hum[1])
    assert abs(d[ls, la] - al) < 1e-5 and np.count_nonzero(np.abs(d) > 1e-9) == 1, d
    # 3) gamma 0: only the notes one move from the goal get a value (exactly 1), the rest stays exactly 0
    x, log, moves, qs = l17_run([(10, 'l17-gamma 0'), (10, 'l17-eps 0.5'), (10, 'l17-goal 0'), (30, 'l17-practice 300'), (40, 'l17-dump bang')])
    V0 = qs[0].max(1)
    assert np.allclose(V0[1:], [1, 1, 0, 0, 0, 0, 0], atol=1e-3), V0
    # 4) it plays: a melody with bells, plucks and chords
    x, log, _, _ = l17_run([(10, 'l17-speed 8'), (20, 'l17-play 1')], 4.0)
    s = stats(x)
    assert s['finite'] and s['rms'] > 0.01 and s['peak'] < 1, s
    return (f'{n} traced moves = numpy Q-learning (world, epsilon-greedy choice, update within {worst:.0e}; explored {rate:.3f}, epsilon {eps}); '
            f'after 400 phrases the values = value iteration within {np.abs(V - Vs).max():.3f}; good! adds alpha to that move only; '
            f'gamma 0 values {np.round(V0, 2).tolist()}; plays (rms {s["rms"]:.3f})')


def check_all_load():
    """every help patch (so every abstraction, with arguments), the start page and all lessons load and run without a single error"""
    pats = sorted(os.path.join(PKG, f) for f in os.listdir(PKG) if f.endswith('-help.pd') or f in ('00-START-HERE.pd', 'neuralplay-meta.pd'))
    log = run_pd(pats + [os.path.join(HERE, 'tests', 'quit-after-1s.pd')], timeout=120)
    assert not errors_in(log), errors_in(log)[:5]
    x, log = render('00-START-HERE.pd', 1.0, [(10 + 10 * k, f'neuralplay-open {f}') for k, f in
                                              enumerate(sorted(f for f in os.listdir(os.path.join(PKG, 'lessons')) if f.endswith('.pd')))])
    assert not errors_in(log), errors_in(log)[:5]
    for d in ('examples', 'exercises'):   # the start page browser lists the folder and opens file number k
        fs = folder(d)
        x, log = render('00-START-HERE.pd', 1.0, [(100 + 10 * k, f'neuralplay-{d} {k}') for k in range(len(fs))])
        got = [l.split()[-1] for l in log.splitlines() if l.startswith('neuralplay: opening')]
        assert not errors_in(log) and got == fs, (d, got, fs, errors_in(log)[:3])
    return f'{len(pats)} help/start/meta patches load cleanly; the start page opens all lessons, examples and exercises without errors'


def folder(d):
    return sorted(f for f in os.listdir(os.path.join(PKG, d)) if f.endswith('.pd'))


def check_smoke():
    """every patch in examples/, exercises/ and templates/ runs 3 s: no errors, finite audio; examples and templates make sound"""
    out = []
    for d in ('examples', 'exercises', 'templates'):
        for f in folder(d):
            x, log = render(f'{d}/{f}', 3.0)
            s = stats(x)
            assert not errors_in(log) and s['finite'], (f, errors_in(log)[:3], s)
            assert d == 'exercises' or s['rms'] > 0.001, (f, 'silent', s)
            out.append(f'{f} (rms {s["rms"]:.3f})')
    return ', '.join(out)


def check_exercises():
    """convention: exercises/exNN-*.pd runs its self-check on [r exNN-check], loads the answer on [r exNN-solution]
    and prints 'exNN-solved: 1' or 0. Unsolved must fail, the solution must pass."""
    out = []
    for f in folder('exercises'):
        ex = f.split('-')[0]
        res = []
        for ev in ([], [(10, f'{ex}-solution bang')]):
            x, log = render(f'exercises/{f}', 5.0, ev + [(500, f'{ex}-check bang')])
            assert not errors_in(log), (f, errors_in(log)[:3])
            res += printed(log, f'{ex}-solved')
        assert res == [0, 1], (f, 'expected unsolved 0, solution 1, got', res)
        out.append(f)
    return 'self-checks fail without the answer and pass with the solution: ' + ', '.join(out)



# ----------------------------------------------------------------------------- examples/gan-gesture-space.pd
# Reference math, written independently of the patch: the same model and the same order of operations.
VG = 'examples/gan-gesture-space.pd'
VG_LAYERS = [(8, 4), (4, 2), (4, 2), (2, 4), (4, 8)]          # (fan_in, fan_out) of the 5 layers of a synth preset
VG_D, VG_H, VG_MR, VG_MF = 152, 16, 2, 2                     # one gesture = 4 x 32 FM frames + 24 cloud numbers
VG_GP = 320
VGP = 'examples/vae-feedback-gan-presets.pd'                  # the same GAN on static synth presets (108 numbers)


def vg_off(D):
    """critic layout for items of D numbers: b1, B2, b2 offsets and the total size (B1 starts at 0)"""
    return 16 * D, 16 * D + 16, 16 * D + 32, 16 * D + 33
VG_CPN = [(0, 1, 0), (0, 0.5, 0), (10, 500, 1), (1, 100, 1), (-24, 24, 0), (0, 1, 0)]   # cloud parameters: lo, hi, log


def vg_fan_in():
    return np.concatenate([[i] * (i * o + o) for i, o in VG_LAYERS])


def vg_static_net(cfg):
    """the audio-rate VAE with feedback 0 (x = 0): outputs, mu, log-variance"""
    h = np.tanh(cfg[32:36]); mu = cfg[36:44].reshape(2, 4) @ h + cfg[44:46]; lv = cfg[46:54].reshape(2, 4) @ h + cfg[54:56]
    g = np.tanh(cfg[56:64].reshape(4, 2) @ mu + cfg[64:68])
    return np.tanh(cfg[68:100].reshape(8, 4) @ g + cfg[100:108]), mu, lv


def vg_nat(u, j):
    lo, hi, log = VG_CPN[j]; x = (u + 1) / 2
    return lo * (hi / lo) ** x if log else lo + (hi - lo) * x


def vg_generate(gp, ges, n, z, T=1.0):
    """G(z) = sum_n w_n g_n with w = softmax((A2 tanh(A1 z + a1) + a2) / T)"""
    h = np.tanh(gp[:32].reshape(16, 2) @ z + gp[32:48])
    L = gp[48:304].reshape(16, 16)[:n] @ h + gp[304:304 + n]
    w = np.exp((L - L.max()) / T); w /= w.sum()
    return w @ ges[:n], w, h


def vg_critic(dp, x):
    D = len(x); b1, B2, b2, _ = vg_off(D)
    q = dp[:b1].reshape(16, D) @ x + dp[b1:B2]; a = np.where(q > 0, q, 0.2 * q)
    return dp[B2:b2] @ a + dp[b2], q, a


def vg_dlrelu(q):
    return np.where(q > 0, 1.0, 0.2)


def vg_sig(x):
    return 1 / (1 + np.exp(-x))


def vg_noise(rn, k, D):
    return rn[k * D:(k + 1) * D]


def vg_losses(gp, dp, ges, n, rz, ri, rn, sn):
    D = ges.shape[1]; LD = 0.0
    for m in range(VG_MR): LD -= np.log(vg_sig(vg_critic(dp, ges[int(ri[m])] + sn * vg_noise(rn, m, D))[0])) / VG_MR
    for m in range(VG_MF):
        x = vg_generate(gp, ges, n, rz[2 * m:2 * m + 2])[0] + sn * vg_noise(rn, VG_MR + m, D)
        LD -= np.log(1 - vg_sig(vg_critic(dp, x)[0])) / VG_MF
    LG = 0.0
    for m in range(VG_MF):
        x = vg_generate(gp, ges, n, rz[4 + 2 * m:6 + 2 * m])[0] + sn * vg_noise(rn, VG_MR + VG_MF + m, D)
        LG -= np.log(vg_sig(vg_critic(dp, x)[0])) / VG_MF
    return LD, LG


def vg_dgrad(gp, dp, ges, n, rz, ri, rn, sn):
    D = ges.shape[1]; b1, B2, b2, DP = vg_off(D); dg = np.zeros(DP)
    xs = [(ges[int(ri[m])] + sn * vg_noise(rn, m, D), True) for m in range(VG_MR)] + \
         [(vg_generate(gp, ges, n, rz[2 * m:2 * m + 2])[0] + sn * vg_noise(rn, VG_MR + m, D), False) for m in range(VG_MF)]
    for x, real in xs:
        l, q, a = vg_critic(dp, x); g = (vg_sig(l) - 1) / VG_MR if real else vg_sig(l) / VG_MF
        d = g * dp[B2:b2] * vg_dlrelu(q)
        dg[:b1] += np.outer(d, x).ravel(); dg[b1:B2] += d; dg[B2:b2] += g * a; dg[b2] += g
    return dg


def vg_ggrad(gp, dp, ges, n, rz, rn, sn):
    D = ges.shape[1]; b1, B2, b2, _ = vg_off(D)
    gg = np.zeros(VG_GP); B1 = dp[:b1].reshape(16, D); A2 = gp[48:304].reshape(16, 16)[:n]
    for m in range(VG_MF):
        z = rz[4 + 2 * m:6 + 2 * m]
        x, w, h = vg_generate(gp, ges, n, z); x = x + sn * vg_noise(rn, VG_MR + VG_MF + m, D)
        l, q, a = vg_critic(dp, x); g = (vg_sig(l) - 1) / VG_MF
        dx = B1.T @ (g * dp[B2:b2] * vg_dlrelu(q))
        dw = ges[:n] @ dx; dl = w * (dw - w @ dw)                    # softmax Jacobian
        gg[48:304].reshape(16, 16)[:n] += np.outer(dl, h); gg[304:304 + n] += dl
        dpre = (A2.T @ dl) * (1 - h * h)
        gg[:32] += np.outer(dpre, z).ravel(); gg[32:48] += dpre
    return gg


def vg_adam(p, g, m, v, t, lr):
    m = 0.5 * m + 0.5 * g; v = 0.999 * v + 0.001 * g * g
    return p - lr * m / (1 - 0.5 ** t) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8), m, v


def vg_step(st, ges, n, rz, ri, rn, lr, sn):
    gp, dp, gm, gv, dm, dv, t = st; t += 1
    dg = vg_dgrad(gp, dp, ges, n, rz, ri, rn, sn); dp, dm, dv = vg_adam(dp, dg, dm, dv, t, lr)
    gg = vg_ggrad(gp, dp, ges, n, rz, rn, sn); gp, gm, gv = vg_adam(gp, gg, gm, gv, t, lr)
    return (gp, dp, gm, gv, dm, dv, t), dg, gg


def vg_init(rng, D):
    b1, B2, b2, DP = vg_off(D)
    gp = np.zeros(VG_GP); dp = np.zeros(DP)
    gp[:32] = rng.normal(0, 2, 32); gp[48:304] = rng.normal(0, 0.25, 256)
    dp[:b1] = rng.normal(0, 1 / np.sqrt(D), b1); dp[B2:b2] = rng.normal(0, 0.25, 16)
    return gp, dp


def vg_io(tmp, arrays):
    for name, a in arrays.items(): np.savetxt(os.path.join(tmp, name + '.txt'), np.ravel(a), fmt='%.9g')


def vg_read(tmp, name):
    return np.loadtxt(os.path.join(tmp, name + '.txt')).ravel()


def bessel_j(n, x, m=4000):
    t = np.linspace(0, np.pi, m); f = np.cos(n * t - x * np.sin(t))
    return float(np.sum((f[1:] + f[:-1]) / 2) * (t[1] - t[0]) / np.pi)


QUIET_GRAINS = [(15, 'exvg-granlevel 0')]


def check_vg_presets():
    """random synth presets are LeCun normal and repeatable; the latent z of the VAE is mu + tau * sigma * eps"""
    dirs = [tempfile.mkdtemp() for _ in range(6)]; ev = []
    for k, (seed, d) in enumerate(zip([3, 7, 12, 21, 33, 42], dirs)):
        ev += [(20 + 40 * k, f'exvg-seed {seed}'), (40 + 40 * k, f'exvg-io write {d}')]
    x, log = render(VG, 0.5, ev)
    assert not errors_in(log), errors_in(log)[:3]
    spread = vg_read(dirs[0], 'meta')[4]
    g = np.stack([vg_read(d, 'tgt') for d in dirs]) * np.sqrt(vg_fan_in()) / spread     # back to N(0, 1)
    assert abs(g.mean()) < 0.12 and abs(g.std() - 1) < 0.1, ('not N(0, spread^2 / fan_in)', g.mean(), g.std())
    d2 = tempfile.mkdtemp(); render(VG, 0.3, [(20, 'exvg-seed 3'), (40, f'exvg-io write {d2}')])
    assert np.array_equal(vg_read(d2, 'tgt'), vg_read(dirs[0], 'tgt')), 'the same seed gave a different preset'
    tmp = tempfile.mkdtemp()
    x, log = render(VG, 2.0, QUIET_GRAINS + [(20, 'exvg-glide 0'), (30, 'exvg-seed 3'), (120, 'exvg-fb 0'), (130, 'exvg-tau 1'),
                                               (140, 'exvg-solo 99'), (150, 'exvg-probe 1'), (300, f'exvg-io write {tmp}')])
    _, mu, lv = vg_static_net(vg_read(tmp, 'cfg'))
    z = x[int(0.5 * SR):, 0]; sd = np.exp(lv[0] / 2); kurt = np.mean((z - z.mean()) ** 4) / z.var() ** 2
    assert abs(z.mean() - mu[0]) < 4 * sd / np.sqrt(len(z)) + 1e-3 and abs(z.std() / sd - 1) < 0.02 and abs(kurt - 3) < 0.1, (z.mean(), mu[0], z.std(), sd, kurt)
    return (f'6 seeded presets, standardised mean {g.mean():.3f} std {g.std():.3f}, repeatable; latent z: mean {z.mean():.3f} (mu {mu[0]:.3f}), '
            f'std {z.std():.3f} (tau sigma {sd:.3f}), kurtosis {kurt:.2f}')


def check_vg_fm():
    """with feedback 0 and tau 0 a voice is plain FM: its lines have Bessel amplitudes |J_n(I)| for the index the network gives"""
    tmp = tempfile.mkdtemp(); r = 1 / np.pi
    x, log = render(VG, 3.0, QUIET_GRAINS + [(20, 'exvg-glide 0'), (30, 'exvg-seed 3'), (120, 'exvg-fb 0'), (130, 'exvg-tau 0'), (140, 'exvg-imax 3'),
                                               (150, f'exvg-ratio {r:.6f}'), (160, 'exvg-f0 45'), (170, 'exvg-solo 3'), (400, f'exvg-io write {tmp}')])
    assert not errors_in(log), errors_in(log)[:3]
    I = 3 * (vg_static_net(vg_read(tmp, 'cfg'))[0][2] + 1) / 2
    y = x[int(0.8 * SR):int(2.8 * SR), 0]; w = np.hanning(len(y)); S = np.abs(np.fft.rfft(y * w)); f = np.fft.rfftfreq(len(y), 1 / SR)
    fc, fm = 330.0, 330.0 * r
    ns = [n for n in range(-8, 11) if abs(fc + n * fm) > 60]
    meas = np.array([S[(f > abs(fc + n * fm) - 2) & (f < abs(fc + n * fm) + 2)].max() for n in ns])
    pred = lambda i: np.array([abs(bessel_j(n, i)) for n in ns])
    unit = lambda v: v / np.linalg.norm(v)
    cos = float(unit(meas) @ unit(pred(I)))
    grid = np.linspace(0.05, 6, 1191); fit = grid[np.argmin([np.linalg.norm(unit(meas) - unit(pred(i))) for i in grid])]
    assert I > 1 and cos > 0.995 and abs(fit - I) < 0.02 * I + 0.01, (I, fit, cos)
    return f'voice 3 of the seed-3 preset: the network gives index {I:.3f}, its spectrum fits Bessel lines with index {fit:.3f} (profile match {cos:.4f}, J_0 nearly 0)'


def check_vg_gestures():
    """recording stores each frame of the slider (u = 2 x - 1); playing interpolates the frames linearly, to float precision"""
    d = tempfile.mkdtemp(); t0 = 300
    ev = [(t0, 'exvg-record bang')] + [(t0 + 125 * f + 60, f'exvg-imax {0.2 * f:.3f}') for f in range(31)]
    ev += [(t0 + 4100, f'exvg-io write {d}'), (t0 + 4110, 'exvg-trace 1'), (t0 + 4700, 'exvg-trace 0')]
    x, log = render(VG, 5.2, ev)
    assert not errors_in(log) and any('exvg-recorded' in l for l in log.splitlines()), errors_in(log)[:3]
    cur = vg_read(d, 'cur')
    want = np.array([2 * 3 / 8 - 1] + [2 * (0.2 * f) / 8 - 1 for f in range(31)])
    rec = np.abs(cur[:32] - want).max()
    flat = max(np.ptp(cur[32 * p:32 * (p + 1)]) for p in (1, 2, 3))      # the lanes nobody moved stay constant
    assert rec < 1e-6 and flat < 1e-9, (rec, flat)
    tr = np.array([list(map(float, l.split()[1:])) for l in log.splitlines() if l.startswith('exvg-trace:')])
    pos = tr[:, 0]; f0 = np.floor(pos).astype(int); a = pos - f0
    u0 = cur[f0] + (cur[(f0 + 1) % 32] - cur[f0]) * a
    u1 = cur[32 + f0] + (cur[32 + (f0 + 1) % 32] - cur[32 + f0]) * a
    play = max(np.abs(tr[:, 1] - 8 * (u0 + 1) / 2).max(), np.abs(tr[:, 2] - 0.25 * 16 ** ((u1 + 1) / 2)).max())
    assert len(tr) > 20 and play < 1e-4, (len(tr), play)
    return f'32 frames recorded exactly (within {rec:.0e}); {len(tr)} playback ticks = linear interpolation of the frames (within {play:.0e})'


def check_vg_granular():
    """a chirp as the sound: a grain at position p, pitch k sounds at (200 + 1600 p_centre) * 2^(k/12) Hz; the density is exact"""
    out = []
    def clouds(pos, size, dens, pitch):
        return [(20, 'exvg-fmlevel 0'), (20, 'exvg-granlevel 1'), (30, 'exvg-chirp 200 1800'), (40, f'exvg-g0 {pos}'), (40, 'exvg-g1 0'),
                (40, f'exvg-g2 {size}'), (40, f'exvg-g3 {dens}'), (40, f'exvg-g4 {pitch}'), (40, 'exvg-g5 1'),
                (40, 'exvg-g11 0'), (40, 'exvg-g17 0'), (40, 'exvg-g23 0')]
    for pos, pitch in [(0.25, 7), (0.75, 0), (0.5, -12)]:
        x, log = render(VG, 3.0, clouds(pos, 100, 20, pitch))
        assert not errors_in(log), errors_in(log)[:3]
        y = x[int(0.8 * SR):int(2.8 * SR), 0]
        L = 192000; span = 2 ** (pitch / 12) * 48000 * 0.1; start = min(max(pos * L, 1), L - span - 2)
        want = (200 + 1600 * (start + span / 2) / L) * 2 ** (pitch / 12)
        S = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2; f = np.fft.rfftfreq(len(y), 1 / SR)
        band = (f > want - 250) & (f < want + 250); cen = (S[band] * f[band]).sum() / S[band].sum()
        assert abs(cen / want - 1) < 0.015, (pos, pitch, want, cen)
        out.append(f'{cen:.0f} Hz (expected {want:.0f})')
    counts = []
    for dens in (10, 40):
        x, log = render(VG, 3.0, clouds(0.5, 15, dens, 0))
        y = np.abs(x[int(1.0 * SR):int(3.0 * SR), 0]); env = np.convolve(y, np.ones(96) / 96, 'same'); on = env > 0.5 * env.max()
        n = len(np.flatnonzero(on[1:] & ~on[:-1])); counts.append(n)
        assert abs(n - 2 * dens) <= 1, (dens, n)
    return 'grain pitch and position: ' + ', '.join(out) + f'; {counts[0]} and {counts[1]} grains in 2 s at density 10 and 40'


def gan_math_check(patch, D, data, walked, extra, data_scale):
    """gradients = finite differences; one Pd training step = numpy; walking z = numpy. Returns (summary, walked array, numpy G(z))"""
    np.seterr(all='ignore')
    f32 = lambda a: np.asarray(a, dtype=np.float32).astype(np.float64)
    rng = np.random.default_rng(5); n = 6; _, _, _, DP = vg_off(D)
    ges = np.zeros((16, D)); ges[:n] = f32(data_scale(rng.normal(0, 0.8, (n, D))))
    gp, dp = vg_init(rng, D)
    draws = lambda: (rng.uniform(-1, 1, 8), rng.integers(n, size=2).astype(float), rng.normal(size=6 * D))
    rz, ri, rn = draws(); sn = 0.05
    dg = vg_dgrad(gp, dp, ges, n, rz, ri, rn, sn); gg = vg_ggrad(gp, dp, ges, n, rz, rn, sn); worst = 0.0
    for vec, ana, which in [(dp, dg, 0), (gp, gg, 1)]:
        for i in rng.choice(np.flatnonzero(np.abs(ana) > 1e-5), 15, replace=False):
            e = 1e-6; vp, vm = vec.copy(), vec.copy(); vp[i] += e; vm[i] -= e
            Lp = vg_losses(vp, dp, ges, n, rz, ri, rn, sn)[1] if which else vg_losses(gp, vp, ges, n, rz, ri, rn, sn)[0]
            Lm = vg_losses(vm, dp, ges, n, rz, ri, rn, sn)[1] if which else vg_losses(gp, vm, ges, n, rz, ri, rn, sn)[0]
            num = (Lp - Lm) / (2 * e); worst = max(worst, abs(num - ana[i]) / (abs(num) + abs(ana[i])))
    assert worst < 1e-4, ('analytic gradient != finite differences', worst)
    st = (gp, dp, np.zeros(VG_GP), np.zeros(VG_GP), np.zeros(DP), np.zeros(DP), 0)
    for _ in range(2):                                   # 2 numpy steps first: Adam moments are not zero, t = 2
        st = vg_step(st, ges, n, *draws(), 0.005, sn)[0]
    st = tuple(f32(a) if isinstance(a, np.ndarray) else a for a in st)
    rz, ri, rn = draws(); rz, rn = f32(rz), f32(rn)
    gp, dp, gm, gv, dm, dv, t = st
    tin, tout, twalk = tempfile.mkdtemp(), tempfile.mkdtemp(), tempfile.mkdtemp()
    vg_io(tin, dict(gp=gp, dp=dp, gm=gm, gv=gv, dm=dm, dv=dv, rz=rz, ri=ri, rn=rn, meta=[t, n, 0, 0, 0], cfg=np.zeros(108), tgt=np.zeros(108),
                    dg=np.zeros(DP), gg=np.zeros(VG_GP), **{data: ges}, **extra))
    x, log = render(patch, 0.5, [(50, f'exvg-io read {tin}'), (60, 'exvg-step bang'), (70, f'exvg-io write {tout}'),
                                 (80, 'exvg-gen 0.3 -0.5 1.5'), (90, f'exvg-io write {twalk}')])
    assert not errors_in(log), errors_in(log)[:3]
    meta = vg_read(tout, 'meta'); lr, sn = meta[2], meta[3]
    (gp3, dp3, gm3, gv3, dm3, dv3, t3), dg3, gg3 = vg_step(st, ges, n, rz, ri, rn, lr, sn)
    errs = {nm: np.abs(vg_read(tout, nm) - ref).max() / max(np.abs(ref).max(), 1e-12)
            for nm, ref in [('dg', dg3), ('gg', gg3), ('dm', dm3), ('dv', dv3), ('gm', gm3), ('gv', gv3)]}
    assert max(errs.values()) < 1e-4 and meta[0] == t3, (errs, meta)
    dpar = max(np.abs(vg_read(tout, 'dp') - dp3).max(), np.abs(vg_read(tout, 'gp') - gp3).max())
    assert dpar < 1e-4, dpar
    want = vg_generate(gp3, ges, n, np.array([0.3, -0.5]), 1.5)[0]
    gerr = np.abs(vg_read(twalk, walked) - want).max()
    assert gerr < 1e-4, gerr
    return (f'gradients = finite differences (worst {worst:.1e}); one Pd step = numpy: gradients and Adam moments within {max(errs.values()):.1e}, '
            f'parameters within {dpar:.1e}; walking z = (0.3, -0.5) at T = 1.5 = numpy (within {gerr:.1e})'), twalk, want


def check_vg_gan_math():
    """the GAN of the gesture space (152 numbers per gesture), and walking also sets the 24 cloud values"""
    msg, twalk, want = gan_math_check(VG, VG_D, 'ges', 'cur', dict(cur=np.zeros(VG_D), gnat=np.zeros(24)), np.tanh)
    nat = np.array([vg_nat(u, k % 6) for k, u in enumerate(want[128:])])
    cerr = np.max(np.abs(vg_read(twalk, 'gnat') - nat) / np.maximum(np.abs(nat), 1))
    assert cerr < 1e-4, cerr
    return msg + ', and its cloud values'


def check_vgp_gan_math():
    """the same GAN on static presets (108 numbers): examples/vae-feedback-gan-presets.pd"""
    return gan_math_check(VGP, 108, 'fav', 'tgt', {}, lambda a: a / np.sqrt(4))[0]


def check_gan_files_safe():
    """WRITE never empties a saved file: with nothing kept it leaves gestures.txt / favourites.txt alone, with the demo set
    it writes it. Runs on copies in a temp folder (the patches write next to themselves), never in the package."""
    out = []
    for patch, fname, nums in [(VG, 'gestures.txt', VG_D), (VGP, 'favourites.txt', 108)]:
        src = open(os.path.join(PKG, patch)).read().replace('\\$0-write empty write', '\\$0-write nntest-write write', 1)
        assert 'nntest-write' in src, 'write button not found'
        tmp = tempfile.mkdtemp(); cp = os.path.join(tmp, 'copy.pd'); f = os.path.join(tmp, fname)
        open(cp, 'w').write(src); open(f, 'w').write('1 2 3;\n')
        log = run_pd([cp], extra=('-path', PKG, '-send', 'nntest-write bang', '-send', 'pd quit'))
        assert open(f).read() == '1 2 3;\n' and 'nothing kept yet' in log, (fname, 'an empty write touched the file')
        run_pd([cp], extra=('-path', PKG, '-send', 'exvg-demo bang', '-send', 'nntest-write bang', '-send', 'pd quit'))
        rows = [r for r in open(f).read().split(';') if r.strip()]
        assert len(rows) == 6 and all(len(r.split()) == nums for r in rows), (fname, len(rows))
        out.append(f'{fname}: an empty write leaves the old file alone, the demo set writes 6 x {nums} numbers')
    return '; '.join(out)


def check_vg_train():
    """trained on the 6 demo gestures for 600 steps (one per 20 ms), the GAN can play several of them (and the patch survives it)"""
    x, log = render(VG, 12.5, [(20, 'exvg-demo bang'), (200, 'exvg-train 1'), (12200, 'exvg-train 0'), (12300, 'exvg-print bang')])
    assert not errors_in(log) and stats(x)['finite'], errors_in(log)[:3]
    found = printed(log, 'exvg-found'); st = [float(v) for v in [l for l in log.splitlines() if l.startswith('exvg-stats:')][-1].split()[1:]]
    assert st[0] > 550 and found and found[-1] >= 3, (st, found)
    return f'{int(st[0])} steps: D(real) {st[1]:.2f}, D(fake) {st[2]:.2f}, the map shows {int(found[-1])} of 6 gestures'


# ----------------------------------------------------------------------------- examples/rl-applause-drummer.pd
EXRL = 'examples/rl-applause-drummer.pd'
EXRL_TASTE = {'house': ('x...x...x...x...', '....x.......x...', '..x...x...x...x.', '.......x......x.'),   # kick snare hat rim
              'clave': ('x.......x.......', '................', 'x.xxx.xxx.xxx.xx', 'x..x..x...x.x...')}


def exrl_taste(name):
    return np.array([[1.0 if ch == 'x' else 0.0 for ch in row] for row in EXRL_TASTE[name]])


def exrl_run(events, seconds=1.0):
    x, log = render(EXRL, seconds, events)
    assert not errors_in(log), errors_in(log)[:3]
    ev = [('step', list(map(float, l.split()[1:]))) if l.startswith('exrl-step:') else ('clap', float(l.split()[1]))
          for l in log.splitlines() if l.startswith(('exrl-step:', 'exrl-clap:'))]
    dump = {}
    for l in log.splitlines():
        m = re.match(r'exrl-(th|e|p)([kshr]):', l)
        if m: dump.setdefault(m.group(1), {})[m.group(2)] = np.array(list(map(float, l.split()[1:])))
    dump = {k: np.stack([v[d] for d in 'kshr']) for k, v in dump.items()}
    return x, ev, dump


def exrl_replay(ev, taste, T, lr, lam, dancer):
    """REINFORCE for 4 Bernoulli-logistic neurons with an eligibility trace, in numpy, on the traced hits and claps"""
    th, e = np.zeros((4, 16)), np.zeros((4, 16)); worst = 0.0; dev = var = 0.0; n = 0
    for kind, v in ev:
        if kind == 'clap':
            th += lr * v * e / T; continue
        t, x, p, rc = int(v[0]), np.array(v[1:5]), np.array(v[5:9]), v[9]
        worst = max(worst, np.abs(p - 1 / (1 + np.exp(-th[:, t] / T))).max(),
                    abs(rc - dancer * np.mean((2 * x - 1) * (2 * taste[:, t] - 1))))
        e = lam * e; e[:, t] += x - p
        th[:, t] += lr * rc * (x - p) / T
        dev += (x - p).sum(); var += (p * (1 - p)).sum(); n += 1
    return th, e, worst, dev / np.sqrt(var), n


def check_rl_drummer_math():
    T, lr, lam = 1.5, 0.8, 0.6
    ev0 = [(5, 'exrl-play 0'), (6, 'exrl-forget bang'), (10, f'exrl-temp {T}'), (10, f'exrl-rate {lr}'), (10, f'exrl-lambda {lam}'), (10, 'exrl-dancer 1'), (10, 'exrl-style clave'),
           (20, 'exrl-trace 1'), (30, 'exrl-play 1'), (1000, 'exrl-clap bang'), (2000, 'exrl-boo bang'), (2470, 'exrl-clap bang'), (3000, 'exrl-play 0'),
           (3010, 'exrl-practice 20'), (3020, 'exrl-dump bang')]
    x, ev, dump = exrl_run(ev0, 3.1)
    th, e, worst, z, n = exrl_replay(ev, exrl_taste('clave'), T, lr, lam, 1)
    claps = sum(k == 'clap' for k, _ in ev)
    err = max(np.abs(dump['th'] - th).max(), np.abs(dump['e'] - e).max(), np.abs(dump['p'] - 1 / (1 + np.exp(-th / T))).max())
    assert n > 330 and claps == 3 and worst < 2e-5 and err < 1e-4 and abs(z) < 4, (n, claps, worst, err, z)
    return (f'{n} traced 16ths and {claps} claps/boos = numpy REINFORCE (probabilities and dancer rewards within {worst:.0e}, '
            f'weights and memory within {err:.0e}); hits follow p (z = {z:+.2f})')


def check_rl_drummer_learns():
    out = []
    for style in ('house', 'clave'):
        x, ev, dump = exrl_run([(10, f'exrl-style {style}'), (20, 'exrl-practice 120'), (30, 'exrl-dump bang')], 0.5)
        want = exrl_taste(style) > 0.5; p = dump['p']
        assert np.array_equal(p > 0.5, want), (style, np.round(p, 2))
        out.append(f'{style}: all 64 cells right (p {p[want].min():.2f}..{p[want].max():.2f} on hits, at most {p[~want].max():.2f} on rests)')
    x, _, _ = exrl_run([], 3.0)
    s = stats(x)
    assert s['finite'] and s['rms'] > 0.01 and s['peak'] <= 1, s
    return 'after 120 bars with the dancer the policy plays its taste - ' + '; '.join(out) + f'; plays at load (rms {s["rms"]:.3f})'


CHECKS = [('everything loads', check_all_load), ('nn.act~ formulas', check_act), ('nn.neuron~ math', check_neuron), ('nn.izh~ spikes', check_izh), ('nn.act / nn.dense / nn.softmax', check_ctl),
          ('lesson 01 logic rhythms', check_l01),
          ('lesson 02 activation waveshaping', check_l02),
          ('lesson 03 neuron anatomy', check_l03),
          ('lesson 04 perceptron learning', check_l04),
          ('lesson 05 hidden layer XOR', check_l05),
          ('lesson 06 backprop', check_l06),
          ('lesson 07 Hopfield memory', check_l07),
          ('lesson 08 network as mapping', check_l08),
          ('lesson 09 CTRNN oscillator', check_l09),
          ('lesson 10 spiking neurons', check_l10),
          ('lesson 11 autoencoder / VAE', check_l11),
          ('lesson 12 GAN', check_l12),
          ('lesson 13 attention', check_l13),
          ('lesson 14 induction head', check_l14),
          ('lesson 15 diffusion', check_l15),
          ('lesson 16 feedback reservoir', check_l16),
          ('lesson 17 Q-learning', check_l17),
          ('examples / exercises / templates run', check_smoke),
          ('exercise self-checks', check_exercises),
          ('gesture-gan: presets and VAE sampling', check_vg_presets),
          ('gesture-gan: FM voice = Bessel lines', check_vg_fm),
          ('gesture-gan: gestures record and play', check_vg_gestures),
          ('gesture-gan: grain pitch, position, density', check_vg_granular),
          ('gesture-gan: GAN math = numpy', check_vg_gan_math),
          ('preset-gan: GAN math = numpy', check_vgp_gan_math),
          ('GAN examples: write never empties a saved file', check_gan_files_safe),
          ('gesture-gan: GAN learns the gestures', check_vg_train),
          ('rl-drummer: REINFORCE math = numpy', check_rl_drummer_math),
          ('rl-drummer: learns the dancer\'s taste', check_rl_drummer_learns)]

if __name__ == '__main__':
    flt = sys.argv[1] if len(sys.argv) > 1 else ''
    fails = 0
    for name, fn in CHECKS:
        if flt not in name: continue
        try:
            print(f'PASS  {name}: {fn()}')
        except Exception as e:
            fails += 1; print(f'FAIL  {name}: {e}')
    sys.exit(1 if fails else 0)
