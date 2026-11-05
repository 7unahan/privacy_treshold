# Wrong Input, Wrong Threshold

Measurement code for *Wrong Input, Wrong Threshold: Re-evaluating Adversarial
Facial Privacy with a Threshold-Free Measure*.

Published adversarial facial-privacy methods report protection above 90%
against black-box recognizers. That number comes from an evaluation protocol
with two independent defects, and the unit it is reported in cannot size
either one.

- **The input is wrong.** Methods published since 2022 feed the recognition
  network the whole photograph, or at most a detector's bounding box.
  Deployed systems detect, align to a landmark template, and crop. Re-scoring
  the same protected files through a realistic front end, regenerating
  nothing, costs a standard transfer attack 56% of its measured dodging
  protection and a budget-allocating attack 7%. The cause is a resampling
  mismatch, not perturbation lost outside the crop.
- **The thresholds are wrong.** Four constants inherited across this
  literature since 2021 hold the false-accept rate near its nominal 1% on
  aligned crops but admit 2.3-5.4% on whole images, with a bias that is
  *opposite-signed* across dodging and impersonation.
- **The measure is wrong.** Both of the above are usually reported as a
  difference of two success rates, which saturates at both ends of its range.
  Over a budget sweep that difference moved by a factor of 8.2 while the
  effect beneath it moved by 1.34.

This repository exists so that none of the three has to be taken on trust.

## What is here

| path | what |
|---|---|
| `src/dcos.py` | the threshold-free measure, with the algebraic self-check |
| `src/constants.py` | every measurement constant, in one place |
| `survey/front_ends.csv` | Table I as data: what 13 released codebases do at evaluation time, with the commit each was read at |
| `docs/REPRODUCTION.md` | which script produces which table and figure |

Still to land: the two front-end operators, the threshold calibration script,
the pipeline-aware remedy, and the stored measurement outputs. See
`docs/REPRODUCTION.md` for the current state.

## The measure, in one minute

```python
import numpy as np
from src.dcos import dcos, identity_check, bootstrap_ci

# cos_dir, cos_ali: PAIRED cosines of the same protected files, scored
# through the whole-image front end and through a deployed one, each
# against the file's own clean identity.
identity_check(cos_dir, cos_ali, tau_dir, tau_ali)   # no threshold may leak in
mean, lo, hi, n = bootstrap_ci(dcos(cos_dir, cos_ali))
```

`Delta cos = cos_dir - cos_ali`. Success means the protected photograph no
longer matches its own identity, `sim < tau`, so a lower cosine is better
protection. Therefore **`Delta cos < 0` means the whole-image protocol
overstates protection**, and `> 0` means it understates it.

Run `python src/dcos.py` for a self-test on synthetic data. It needs no
dataset and no model weights.

## Why there is a self-check in a twenty-line function

The gap can be written two ways: directly as `cos_dir - cos_ali`, or through
the margins with both calibrations still in it. The two are algebraically
identical, so if they ever disagree, a threshold has leaked into a
measurement that is supposed to be free of one, or the two arms are not
paired. `identity_check` computes both and raises. On the 35 stored runs
behind the paper the difference was exactly zero in double precision.

Two of the three errors we caught in our own work were silent: a key
collision that made four folders overwrite each other while the sample count
stayed right, and a resampling mismatch whose mean looked correct while its
peak was three times the nominal budget. Both were found by a check that
compared a new measurement against something already known. That is why the
checks ship with the code rather than living in a notebook.

## What is deliberately not here

- **No face images, protected or clean.** Redistributing identifiable faces
  would contradict the subject of the paper. The stored measurements
  (cosines, thresholds, bootstrap intervals) reproduce every number without
  them.
- **No datasets.** FFHQ, CelebA-HQ and LADN carry their own licences; see
  `docs/REPRODUCTION.md` for where to get each.
- **No model weights.** Source and checksum are given instead.
- **No credentials.** Commercial endpoints are reached through environment
  variables; aggregate results only are published.

## Scope

This measures protections; it does not defeat them. The aligned front end
here reproduces what a deployed recognition pipeline already does, so a
protection that survives it is stronger than one that does not. The point of
publishing the pipeline is that a protection should be evaluated against the
system it claims to protect against.

## Citation

See `CITATION.cff`. The DOI will be added when the paper is published.

## Licence

MIT for the code, CC BY 4.0 for `survey/front_ends.csv` and the figures.
Third-party components are listed in `NOTICE.md`.
