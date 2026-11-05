# Reproduction map

Which script produces which number. Rows marked **pending** are not in this
repository yet; the column says what has to be written or cleaned up before
they can land.

## State

| piece | status |
|---|---|
| threshold-free measure + self-check (`src/dcos.py`) | **in** |
| measurement constants (`src/constants.py`) | **in** |
| survey of 13 released codebases (`survey/front_ends.csv`) | **in** |
| the two front-end operators `P_dir` and `P_ali` | pending |
| threshold calibration at FAR=1% per front end | pending |
| pipeline-aware remedy (differentiable alignment operator) | pending |
| budget and regression checks | pending |
| stored measurement outputs | pending |

## The two front ends

`P_dir` resizes the whole photograph to the recognizer's input resolution.
This is what the surveyed literature evaluates on.

`P_ali` runs a detector, estimates five landmarks, solves a similarity
transform (Umeyama) onto the ArcFace template, warps and crops. This is what
a deployed system does. The aligned crop samples the face about 1.28x larger
than the whole-image resize, which is the whole mechanism: a perturbation
tuned to one sampling grid is attenuated on the other.

Two detectors are supported so that no result rests on one of them.

## Thresholds

Do not reuse a constant. Calibrate on identity-labelled impostor pairs,
**through the front end you are going to evaluate with**, and print the
threshold you used. The four inherited values in `src/constants.py` are
there to reproduce published numbers, not to score new ones.

## Tables and figures

| paper | produced by | notes |
|---|---|---|
| Table I (survey) | `survey/front_ends.csv` | read from code at the listed commits |
| Table II (budget and dataset sweeps) | pending | MI-FGSM, FFHQ N=300 and three datasets N=150 |
| Table III (K1 per protection) | pending | both front ends, thresholds calibrated per pipeline |
| Table IV (inherited thresholds) | pending | false-accept rate of the four constants on each front end |
| Table V (identity gallery) | pending | re-scoring against other photographs of the same subject |
| Table VI (three unseen modern recognizers) | pending | no attack used these as a surrogate |
| Figure 1 (teaser) | pending | |
| Figure 2 (two mechanisms) | pending | frequency panel and scale panel |

## Data

Not redistributed. Each dataset carries its own licence.

| dataset | where | note |
|---|---|---|
| FFHQ (256x256) | NVlabs FFHQ | CC BY-NC-SA 4.0. Already 256x256, no resampling on load. |
| CelebA-HQ | CelebA-HQ release | 1024x1024. **Resize with BILINEAR.** |
| LADN | LADN release | **Resize with BILINEAR.** |

The resampling filter is not a detail. The stored protected images for
CelebA-HQ and LADN were generated from BILINEAR-resized inputs; loading them
with BICUBIC puts the measured sup-norm at 51-65/255 against a nominal 16 and
26, in 598 of 600 images. BILINEAR puts it at 16.000006 and 26.000006 with
zero exceedances. Any script that reads protected images from disk should
print max|delta| against the nominal budget on the first batch, and question
its own loader before it questions the data.

## Models

Weights are not redistributed. Sources and checksums will be listed here.

Four attacked recognizers: IR152, IRSE50, FaceNet, MobileFace.
Three unseen modern recognizers, used as targets only: w600k_r50, w600k_mbf,
glintr100.

## Environment

Runs on Colab and Kaggle from the same file. Detect the platform from
`KAGGLE_KERNEL_RUN_TYPE` / `COLAB_RELEASE_TAG`, never from the presence of a
folder or a package: recent Colab images ship an empty `/kaggle/input`, and
the Kaggle image has `google.colab` installed. Both traps have been hit.
