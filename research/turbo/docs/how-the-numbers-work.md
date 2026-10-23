# How the numbers work

A plain-language tour of every measurement on the results page: what it computes, why it's a fair test, and where it can mislead.

## The ground rule: always compare against the original

Every number on the results page is a *difference*, never an absolute score. For each prompt and random seed, we first let the original model run its full 50 steps (with the faint stripes removed). That image is the **reference**, *R*. Then we generate the same prompt and seed with a fast method, giving the candidate image *C*, and measure how far *C* drifts from *R*.

Why differences? Because some things are *supposed* to repeat. A brick wall, a knitted sweater and a herringbone jacket are genuinely regular. If we scored "repetitiveness" in absolute terms, an accurate picture of a knit sweater would look like a failure. Asking "does the fast version repeat *more than the original did*?" keeps us honest.

## Stamping: the "repeated texture" score

This is the number behind "stamping at the reference's level". The idea is simple: in a stamped image, little pieces of texture are near-copies of each other (the same daisy over and over), while in a natural image every piece is a bit different.

**Step 1: cut the image into small tiles and describe each one.** We split the image into 14×14-pixel patches and run them through DINOv2, a public vision network trained to describe image content. Each patch *i* becomes a list of 384 numbers, a *feature vector* **f**_*i*. Two patches that look alike get similar vectors. We scale each vector to length 1, so similarity is just the dot product, the *cosine similarity*:

```
sim(i, j) = fi · fj = cos θij   (1 = identical, 0 = unrelated)
```

**Step 2: skip the boring patches.** Flat sky or blurry background would look "similar" to each other for an uninteresting reason. So we only keep patches whose brightness actually varies, meaning a standard deviation above 0.02 on a 0–1 scale.

**Step 3: find each patch's closest look-alike, but not its neighbour.** Neighbouring patches naturally resemble each other (the same leaf continues into the next tile). So for each patch we search only patches at least 56 pixels away, and record the best match:

```
nni = max j : dist(i, j) ≥ 56 px  sim(i, j)
```

**Step 4: count the near-duplicates.** A patch whose distant look-alike scores 0.85 or higher is effectively a copy. The **duplicate fraction** is the share of textured patches that have one:

```
dup(image) = (number of patches with nni ≥ 0.85) ÷ (number of textured patches)
```

**Step 5: subtract the original.** The reported "stamping" number is the *excess*:

```
stamping excess = dup(C) − dup(R)
```

So "≈ 0" doesn't mean the image has no repetition. It means the fast image has **no more near-copies than the original model's own image**. Here's a concrete case, the bread at seed 101: the new turbo's excess is 0.003 and `quality`'s is 0.037. In plain words, about 3.7% more of `quality`'s crust patches are near-copies of a distant patch than in the original, against 0.3% for the new turbo. A slightly negative value (−0.001, say) just means a touch *less* copying than the original, which is within noise.

On the results page, "−0.006 for the held-out materials and nature" is the average of that excess over 8 prompts:

```
mean excess = (1/8) Σk=1..8 [ dup(Ck) − dup(Rk) ]
```

**Did we check that this measure means anything?** Yes, and it's the most important part. We collected 15 real photographs from Wikimedia (lichen on rock, pine bark, crumpled paper, tweed, bread crust) and asked: do real photos and the original model score about the same, while the known-stamped turbo scores clearly higher? For **lichen, bark and bread** the answer was yes. Those are the "calibrated" textures, and the numbers I trust most. For **paper and tweed** it was not: real photos also scored high, because paper creases and tweed weave are legitimately repetitive. So I don't use the score on those, and I treat oak and clouds as indicative only.

> Limits: one number per image, a handful of images per category, and a threshold (0.85) chosen by calibration. It can miss some stamping. On seed 202 it gave the old turbo's lichen rosettes a low score, which is why every result was also checked by eye.

## Fine detail: how much "crispness" survives

Fast methods often look waxy because fine detail gets smoothed away. We measure fine detail on the model's internal image (its 128-channel latent, before conversion to pixels). Take the latent *z*, blur it with a 5×5 box average to get *z̄*, and look at what the blur removed. That leftover is the fine detail. Its share of the total energy is the **high-frequency fraction**:

```
HF(z) = mean[ (z − z̄)² ] ÷ mean[ z² ]
```

Again we report the difference from the original, HF(*C*) − HF(*R*). Zero means "as crisp as the original", negative means "smoother (waxier)", and positive means "grittier". The new turbo's −0.004 against `quality`'s −0.022 says `quality` loses noticeably more fine detail.

## Pixel difference: how close is the whole picture?

Convert both images to grayscale brightness *L* (0 = black, 1 = white), using the standard weights for how bright red, green and blue look to us:

```
L = 0.299·red + 0.587·green + 0.114·blue
```

Then average the absolute difference over every pixel:

```
pixel difference = (1/N) Σpixels | LC − LR |
```

0.090 means the new turbo's pixels are, on average, 9% of the full brightness range away from the original's. That sounds large, but this number is dominated by **composition**: if a person stands 30 pixels to the left, every pixel of them "differs" even though the picture is equally good. It's useful for ranking methods (0.090 vs 0.124), not as a quality grade. We chose "same character" over "identical to the original", so some composition drift is expected and fine.

## Brightness

The simplest one: average brightness of the candidate minus the reference, mean(*L*_*C*) − mean(*L*_*R*). `quality`'s −0.049 (and −0.08 on the test set) is the darker, moodier look you noticed. The new turbo's +0.011 is within normal variation.

## Text accuracy

We run Apple's built-in text recognition (the one Photos uses) on each image, then compare what it read with what the prompt asked for. We use a **"bag of characters"** score. Count each character the prompt wanted, and see how many the recognizer found:

```
text score = Σcharacters c min( wanted(c), found(c) ) ÷ total wanted characters
```

1.0 means every wanted character appeared somewhere. It's forgiving about order, which is a real weakness: a ghosted headline with the right letters in the wrong places can still score 0.96. That's exactly what happened with the bakery sign, so **every text image was also checked by eye**. For vertical or brush-style Chinese the recognizer often fails even on the original model's image (score 0.22 or 0.00 on the reference too), so those were judged by eye only.

## Speed

Each "pass" is one run of the 5-billion-parameter network over the image, about 7.8 s on your Mac's 8-bit version at 1 MP. The new turbo uses 15 passes:

```
5 student steps × 1 pass + 1 guided step × 2 passes + 4 finish steps × 2 passes = 15 passes
```

Guided steps cost two passes because classifier-free guidance runs the network twice (with and without the prompt) and blends the results. Measured: 116 s of sampling, about 119 s with text encoding and decoding.

## How confident should you be?

| Evidence | Size | Strength |
|---|---|---|
| Test set, seed 101 | 18 prompts | Same prompts used to develop the method, so some risk of tuning to them |
| Test set, seed 202 | 6 prompts | Different noise, same prompts |
| Held-out prompts | 32 prompts, 9 sizes | Never seen in training or tuning; the strongest evidence |
| Slot experiment (where to put the guided step) | 7 prompts, 1 seed | Clear for the bakery headline; confirmed on held-out text but still a small sample |

In short: the stamping and detail results are consistent across every set and match what the eye sees, so I'm confident in them. The text result rests on fewer examples, so treat "on par with `quality`" as likely rather than proven. That's one reason I suggested stage 3, which trains with more weight on text.
