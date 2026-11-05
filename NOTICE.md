# Third-party components

This repository reproduces an evaluation pipeline. The pipeline is built from
components published by others, and nothing here claims them.

| component | role | source |
|---|---|---|
| ArcFace 5-point template | the canonical landmark target the aligned front end warps to | Deng et al., ArcFace |
| MTCNN | detector and landmark estimator in the aligned front end | Zhang et al. |
| RetinaFace | second detector, used to show the result is not detector-specific | Deng et al. |
| facenet-pytorch | MTCNN implementation | github.com/timesler/facenet-pytorch |
| IR152, IRSE50, FaceNet, MobileFace | the four recognizers this literature attacks | weights distributed by the respective authors |
| InsightFace model zoo | the three modern recognizers used as unseen targets | insightface |

Model weights are not redistributed here. `docs/REPRODUCTION.md` gives the
source and a checksum for each.

The four inherited verification thresholds reproduced in `src/constants.py`
are quoted from released code for the purpose of reproducing published
numbers, and are documented in the paper as unfit for the pipeline they are
applied to.
