# neuralplay — a neural network playground for Pure Data

**by Moisés Horta Valenzuela / SEMILLA.AI Studio**

Build, hear and understand neural networks — from the 1943 logic neuron to VAEs,
GANs, transformers and diffusion — and use them to control synthesis and feedback
systems. Sixteen commented lessons plus a small library of neuron objects, written
for music technology students. Everything is **vanilla Pd (0.54 or newer)**, no externals.

![start page](screenshots/00-START-HERE.jpg)

## Getting started

1. Install with deken (*Help → Find externals*, search `neuralplay`), or copy this folder anywhere.
2. Open **`00-START-HERE.pd`** and click a lesson.
3. In every lesson raise the volume of `[output~]` (it also switches audio on — start low),
   read **THE IDEA**, play with **PLAY**, then follow **TRY THIS**. Every `[pd ...]` box is
   commented: open it to see how it works.

To use the objects in your own patches: `[declare -path neuralplay]`.

## The lessons

| # | lesson | the idea | what you hear |
|---|--------|----------|---------------|
| 01 | McCulloch–Pitts (1943) | a neuron as a logic gate | three rhythms in, one bell decides |
| 02 | Activation functions | the curve of a neuron is a waveshaper | saturation, folding, octave-up, your own drawn curve |
| 03 | The neuron | weights + sum + bias + curve, built twice | a non-linear mixer of a chord |
| 04 | Perceptron (1958 / 1969) | learning from examples; why XOR fails | bells, buzzes and thuds as it learns |
| 05 | Hidden layers | two borders solve XOR | XOR of two rhythms; sped up = 1-bit ring modulation |
| 06 | Backpropagation (1986) | gradient descent through layers | the answers turn into the rhythm 0 1 1 0, the error hiss fades |
| 07 | Hopfield (1982) | memory as an energy landscape | a corrupted groove heals itself |
| 08 | Networks as mappings | 2 knobs → 4 synth parameters | a random-network FM instrument |
| 09 | Recurrent neurons (CTRNN) | leaky neurons in a loop oscillate | from LFO to audio-rate tone |
| 10 | Spiking neurons | LIF (1907) and Izhikevich (2003) | spikes as rhythms and pitches, pinging filters |
| 11 | Autoencoders / VAE | a learned latent space | gliding between timbres, sampling variations |
| 12 | GAN | forger vs critic | the forger learns the teacher's melody style (left/right ear) |
| 13 | Attention | softmax lookup with a temperature | a query morphing between four sounds |
| 14 | Transformer induction head | in-context copying | the network continues your melody |
| 15 | Diffusion | denoising step by step | a noise cloud condenses into a chord |
| 16 | Capstone: feedback reservoir | a 27-neuron network with feedback drives a synth | the NeuralNetwork-VAE-Dense instrument, rebuilt |

![perceptron](screenshots/04-perceptron.jpg)
![latent space](screenshots/11-autoencoder-vae.jpg)
![diffusion](screenshots/15-diffusion.jpg)

## The library

| object | what it is |
|--------|-----------|
| `[nn.neuron~]` | one neuron at audio rate: 5 inputs, weights, bias, activation, optional leak (`tau`), repeatable random weights, net-wide messages (`[s nn-<net>]`) |
| `[nn.act~]` | 14 activation functions as waveshapers (linear, step, sign, sigmoid, tanh, relu, leaky, softsign, sin, gauss, elu, gelu, swish, hardclip) |
| `[nn.act]` | the same curves for numbers |
| `[nn.dense]` | a whole layer on arrays — runs trained weights (see lesson 11) |
| `[nn.softmax]` | scores → probabilities with a temperature |
| `[nn.izh~]` | Izhikevich spiking neuron with 7 presets, from rhythm to audio rate |
| `[nn.plot2d]` | decision map of a 2-input network |
| `[nn.scope~]`, `[nn.spectrum~]` | tiny oscilloscope and spectrum analyser |

Every object has a help patch (right-click → Help).

## Examples and exercises

| patch | what it is |
|-------|-----------|
| `examples/ring-oscillator-drone.pd` | three neurons in a ring, each silencing the next: interlocked LFOs or a buzzing three-phase tone |
| `examples/spiking-drum-machine.pd` | two Izhikevich neurons play kick and hi-hat — no sequencer, no clock |
| `exercises/ex01-a-and-not-b.pd` | program one neuron to play A AND NOT B, then press CHECK |

The start page lists both folders by itself (the MORE panel): drop your own patches in and press *refresh*.

![ring oscillator drone](screenshots/ring-oscillator-drone.jpg)

## Make your own

Copy `templates/lesson-template.pd`, rename it and replace the texts in CAPITALS — it has the
layout, the credit line, `[declare -path ..]` and the naming conventions built in. An
exercise checks itself: `[r exNN-check]` runs the check and prints `exNN-solved: 1` or `0`,
`[r exNN-solution]` loads the answer (see `ex01`). The full guide, with the Pd pitfalls to
avoid, is `CONTRIBUTING.md` in the source repository.

## Train your own latent space

`python/train_vae.py` (numpy only, a few seconds) trains the VAE of lesson 11 on 8 timbres and
writes its weights to `lessons/data/vae/`. Change the timbres, run it, reopen the lesson: Pd
plays your latent space. It is the same split as RAVE and `[nn~]`: train elsewhere, play in Pd.

## Verified

Every patch was rendered headlessly and measured against independent Python references
(for example: the perceptron solves AND / OR / NAND at the same epochs as a Python
perceptron, the backprop lesson learns XOR at the same epoch as a numpy implementation, the
CTRNN oscillates at the simulated frequencies, the Pd VAE matches numpy to 0.002, the
diffusion voices land on chord tones). Examples and exercises are tested too: every
exercise must fail as shipped and pass with its solution. The test tools (`dev/check.py`,
`dev/shoot.py`) live in the source repository next to this package and are not needed to use it.

## Publishing on deken (maintainers)

```sh
# with the deken tool installed (https://github.com/pure-data/deken)
deken package --version 0.1.0 neuralplay
deken upload neuralplay[v0.1.0].dek        # needs a puredata.info account
```

The package is pure Pd (no compiled externals), so one upload works on every platform.

## Credits and license

neuralplay © 2026 Moisés Horta Valenzuela / SEMILLA.AI Studio — BSD 3-Clause (see `LICENSE.txt`).
Lesson 16 is a rebuild of the original *NeuralNetwork-VAE-Dense* patch.
