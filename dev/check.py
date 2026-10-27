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
          ('examples / exercises / templates run', check_smoke),
          ('exercise self-checks', check_exercises)]

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
