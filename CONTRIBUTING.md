# Extending neuralplay

How to add lessons, examples, exercises and library objects so they look like the rest of
the playground, show up on the start page and are tested automatically.

Requirements: Pd vanilla 0.54 or newer. For the tests: Python 3 with numpy. For the
screenshots: macOS (`swiftc`, Screen Recording permission for the terminal).

## Every patch

Start from **`neuralplay/templates/lesson-template.pd`**: copy it, rename it, replace every
text written in CAPITALS. It already has the conventions built in:

- **Layout**: a dark title banner, then the panels **THE IDEA** (story, formula, the link to
  sound), **PLAY** (the network and its controls), **LISTEN** (the voices and `[output~]`) and
  **TRY THIS** (three numbered things to try). The credit line
  `neuralplay - Moisés Horta Valenzuela / SEMILLA.AI Studio` sits in the footer.
- **`[declare -path ..]`** in the footer: patches inside `lessons/`, `examples/`,
  `exercises/` or `templates/` find the `nn.*` objects one folder up.
- **Comment everything.** Every `[pd ...]` subpatch starts with a comment saying what it
  does; steps inside are numbered `1)`, `2)`, ...
- **`[output~]` volume starts at 0.** Raising it also switches DSP on.
- **Global names start with the patch id** (`l17-...`, `ex02-...`) so other patches and the
  tests can drive them; everything else is local (`$0-...`). A net name (the 2nd argument
  of `[nn.neuron~]`) should be unique too: `[s nn-<net>]` reaches every neuron of that net.
- **Arrays that `[expr]` reads** have names that start with a letter and contain no `-`:
  `x$0_in`, not `$0-in`.

## Adding an example

1. Save it as `neuralplay/examples/<what-it-is>.pd` (lowercase, no spaces).
2. On the start page, press *refresh* in the MORE panel: it is listed, numbered and opens with OPEN.
3. `python3 dev/check.py examples` — it must load without errors, produce finite audio and
   make sound (rms > 0.001) within 3 seconds, with no volume raised.

## Adding an exercise

Name it `neuralplay/exercises/exNN-<what-it-is>.pd`; `exNN` is its id. An exercise checks
itself, and the test suite checks the check:

| receive / print | what it does |
|-----------------|--------------|
| `[r exNN-check]` | runs the check (the CHECK button sends here too) |
| `[print exNN-solved]` | prints `exNN-solved: 1` when solved, `0` when not |
| `[r exNN-solution]` | loads the answer (also clickable inside `[pd solution]`) |

`dev/check.py` opens each exercise twice and sends `exNN-check` after 0.5 s: once as it
ships (it must print `0`), once after `exNN-solution` (it must print `1`). The check has to
finish within 4 seconds. Copy `[pd check-my-answer]` from `ex01-a-and-not-b.pd`: it samples
the signals every 10 ms for 3 seconds and counts the moments where the student's network
differs from the target.

Feed the student's network and the check from the **same DSP chain** (direct wires, as
`[pd rhythms]` in ex01 does). A `[r~]` between the source and the network can make the
network run one block (64 samples) behind what the check sees, and a correct answer then
fails at every edge.

## Adding a lesson

1. Save it as `neuralplay/lessons/NN-<topic>.pd`.
2. Add it to the start page. In `00-START-HERE.pd` duplicate the button of lesson 16 and its
   description, set the copy's label to the file name and its send-symbol to `$0-open16`
   (lesson *k* sends to `$0-open<k-1>`). Open `[pd open-a-lesson]`, duplicate
   `[r $0-open15]` and the message under it, change them to `[r $0-open16]` and
   `[NN-<topic>.pd(`, and connect the message to `[t a b]` like the others.
   (Quicker, from any patch: a message box `[; neuralplay-open NN-<topic>.pd(`.)
3. Add a row to the lessons table in `neuralplay/README.md` and the root `README.md`.
4. Write a test (next section). Every lesson is already opened once by the
   `everything loads` check, so load errors are caught even without one.

## Adding a test

`dev/check.py` runs Pd with `-batch` (faster than real time, no sound card). It puts
`dev/shadow/` first on the search path, so every `[output~]` becomes a recorder that captures
the signal *before* the volume. A check is a function that raises on failure and returns a
one-line summary:

```python
def check_l17():
    # open the lesson, record 5 s, and at 10 ms send "1" to the receive name l17-play
    x, log = render('lessons/17-your-topic.pd', 5.0, [(10, 'l17-play 1')])
    assert not errors_in(log), errors_in(log)[:3]          # no Pd errors in the console
    s = stats(x)                                            # rms, peak, dc, finite
    assert s['finite'] and s['rms'] > 0.01, s
    hz = spectrum_peak(x)                                   # strongest frequency
    assert abs(hz - 220) < 5, hz
    return f'plays 220 Hz (rms {s["rms"]:.3f})'
```

Add it to the `CHECKS` list at the bottom with a name (`('lesson 17 your topic', check_l17)`).
Values printed with `[print tag]` come back with `printed(log, 'tag')`. Measure what the lesson
claims against an independent Python reference whenever there is one (see `check_l04`,
`check_l06`, `check_l11`).

Run one check: `python3 dev/check.py "lesson 17"`. Run all before every commit.

## Adding a library object

1. `neuralplay/nn.<name>.pd` (with `~` if it processes audio) and `neuralplay/nn.<name>-help.pd`.
   The help patch shows every inlet, outlet, argument and message, and ends with the credit line.
2. The `everything loads` check opens every `*-help.pd` automatically.
3. List it in the LIBRARY panel of `00-START-HERE.pd` and the library table of `neuralplay/README.md`.

## Screenshots

```sh
python3 dev/shoot.py --package                    # all, into neuralplay/screenshots (a few minutes)
python3 dev/shoot.py --package 04-perceptron ex01 # only names containing these texts
```

It starts its own Pd with a GUI (no audio device), waits, captures each window and quits —
other Pd instances are left alone. `DRIVERS` in `shoot.py` sends messages to a lesson first so
the picture shows it working.

## Releasing on deken

1. Bump `VERSION` in `neuralplay/neuralplay-meta.pd` and the commands in `neuralplay/README.md`.
2. `python3 dev/check.py` — everything PASS.
3. Package and upload (needs the [deken](https://github.com/pure-data/deken) tool and a puredata.info account):

```sh
deken package --version 0.2.0 neuralplay
deken upload "neuralplay[v0.2.0].dek"
```

The package is pure Pd, so one upload works on every platform.

## Pd pitfalls met while building this

- **Comments are re-read as atoms.** `1.` shows as `1`, `04` as `4`, `+1` stays `+1`; a
  number that ends a sentence loses its period; `;` becomes a line break. Number steps as
  `1)` and rephrase sentences that end in a number.
- **`[s~]`/`[r~]` can add one block of delay** depending on the order Pd sorts the DSP
  chain; wire directly when two signals must stay sample-aligned. They also need the same
  block size: leave a `[block~ 1]` subpatch through `[outlet~]`.
- **Fan-out order is undefined.** When one outlet feeds several places and order matters,
  use `[t b f]`, `[t a a]`, ... (right outlet first).
- **`[expr]`**: `random()` returns an integer (use `float(random(0, 1000)) / 1000`), `^` is
  bitwise XOR (use `pow()`), several expressions separated by `;` output right to left.
- **`$0` in message boxes**: don't rely on it; send through `[s $0-name]` objects.
- **Identical neurons stay identical.** A symmetric network that starts from equal states
  never leaves them — add a whisper of `[noise~]` (see the ring-oscillator example).
- **GUI details**: a number box with a receive name has no inlet; number-box labels cannot
  contain spaces; a slider's saved start value snaps to whole pixels (150 becomes 150.141);
  array graphs do not clip, so values outside the range draw outside the box.
- **Headless runs**: switch DSP on yourself (`; pd dsp 1`), and record with
  `[tabwrite~]` + `[soundfiler]` — `[writesf~]` can lose the end of the file when batch mode quits.
