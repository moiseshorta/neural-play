#!/usr/bin/env python3
"""Screenshot Pd patches (macOS). Opens them in a separate Pd GUI with no audio
device, DSP on, waits, then captures every patch window with `screencapture -l`.
Usage: python3 dev/shoot.py out_dir patch.pd [patch.pd ...]   (needs Screen Recording permission)
       python3 dev/shoot.py --package [name-filter ...]       (refresh the README screenshots)
"""
import os, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
PD = os.environ.get('PD', '/Applications/Pd-0.55-2.app/Contents/Resources/bin/pd')


def winlist():
    exe = os.path.join(tempfile.gettempdir(), 'nn-winlist')
    if not os.path.exists(exe):
        subprocess.run(['swiftc', '-O', os.path.join(HERE, 'winlist.swift'), '-o', exe], check=True)
    out = subprocess.run([exe, 'Pd'], capture_output=True, text=True).stdout
    return [l.split('\t') for l in out.splitlines() if l.count('\t') >= 3]


def shoot(out_dir, patches, wait=5.0, width=1600):
    os.makedirs(out_dir, exist_ok=True)
    before = {w[0] for w in winlist()}
    cmd = [PD, '-noaudio', '-stderr', '-send', 'pd dsp 1']
    for p in patches: cmd += ['-open', os.path.abspath(p)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    time.sleep(wait)
    shots = []
    try:
        for wid, owner, title, _ in winlist():
            if wid in before or not title.endswith('.pd'): continue
            png = os.path.join(out_dir, title[:-3].replace('~', '') + '.png')
            subprocess.run(['screencapture', '-x', '-o', f'-l{wid}', png], check=True)
            subprocess.run(['sips', '-Z', str(width), png], capture_output=True)
            shots.append(png)
    finally:
        proc.terminate()   # only the Pd we started
        log = proc.communicate(timeout=10)[1]
    return shots, log


DRIVERS = {   # messages sent 0.5 s after opening, so the screenshot shows the lesson doing something
    '01-mcculloch-pitts': 'l01-play 1', '04-perceptron': 'l04-gate OR; l04-speed 120; l04-train 1',
    '06-backprop': 'l06-steps 3; l06-train 1', '07-hopfield-memory': 'l07-cue B; l07-corrupt bang; l07-rate 4; l07-recall 1',
    '08-mlp-mapping': 'l08-explore 1', '12-gan': 'l12-steps 40; l12-train 1', '14-transformer-induction': 'l14-gen 1',
    '15-diffusion': 'l15-go bang'}


def shoot_package(pkg, out_dir, wait=7.0, only=()):
    """one Pd per patch -> JPEG (1400 px wide, quality 60) for the README; only = name filters"""
    os.makedirs(out_dir, exist_ok=True)
    patches = [os.path.join(pkg, '00-START-HERE.pd'), os.path.join(pkg, 'nn.neuron~-help.pd')]
    for d in ('lessons', 'examples', 'exercises', 'templates'):
        patches += sorted(os.path.join(pkg, d, f) for f in os.listdir(os.path.join(pkg, d)) if f.endswith('.pd'))
    patches = [p for p in patches if not only or any(o in os.path.basename(p) for o in only)]
    tmp = tempfile.mkdtemp(); made = []
    for p in patches:
        name = os.path.basename(p)[:-3]
        extra = []
        if name in DRIVERS:
            d = os.path.join(tmp, 'driver.pd')
            msg = ' \\; '.join(DRIVERS[name].split('; '))
            open(d, 'w').write(f'#N canvas 0 50 200 100 12;\n#X obj 10 10 loadbang;\n#X obj 10 40 delay 500;\n#X msg 10 70 \\; {msg};\n#X connect 0 0 1 0;\n#X connect 1 0 2 0;\n')
            extra = [d]
        shots, _ = shoot(tmp, [p] + extra, wait=wait, width=1400)
        for png in shots:
            if os.path.basename(png).startswith(name.replace('~', '')):
                jpg = os.path.join(out_dir, os.path.basename(png)[:-4] + '.jpg')
                subprocess.run(['sips', '-s', 'format', 'jpeg', '-s', 'formatOptions', '60', png, '--out', jpg], capture_output=True)
                made.append(jpg)
    return made


if __name__ == '__main__':
    if sys.argv[1:2] == ['--package']:   # --package [name filter ...]: refresh neuralplay/screenshots
        pkg = os.path.join(os.path.dirname(HERE), 'neuralplay')
        print('\n'.join(shoot_package(pkg, os.path.join(pkg, 'screenshots'), only=sys.argv[2:])))
        sys.exit(0)
    shots, log = shoot(sys.argv[1], sys.argv[2:])
    print('\n'.join(shots)); print(log[-3000:], file=sys.stderr)
