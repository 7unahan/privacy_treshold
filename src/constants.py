"""Measurement constants. One place, not thirteen.

These are fixed across every run in the paper. Changing one invalidates the
comparison with every stored result, so they live here and are imported,
never redefined in a script.
"""

# attack
MI_EPS = 16 / 255
MI_ITERS = 50
MI_MU = 1.0
TI_K = 15
TI_SIG = 3.0
DI_P = 0.5
DI_MIN = 224

# budget map
SIGMA_FRQ = 1.5
CAP_MULT = 2.0

# calibration and statistics
N_CAL = 500          # clean images per front end for threshold calibration
FAR = 0.01           # nominal false-accept rate
NBOOT = 5000
SEED = 12345
BATCH = 16

# the four recognizers this literature attacks, and their input sizes
TARGETS = ["IR152", "IRSE50", "FaceNet", "MobileFace"]
INPUT_SIZE = {"IR152": 112, "IRSE50": 112, "FaceNet": 160, "MobileFace": 112}

# the four constants inherited across the literature since 2021.
# Present for reference and for reproducing published numbers ONLY.
# Do not use them to score a pipeline you have not calibrated: on whole
# images they admit 2.3-5.4% false accepts against a nominal 1%.
INHERITED_THRESHOLDS = {
    "IR152": 0.166788,
    "IRSE50": 0.241045,
    "FaceNet": 0.409131,
    "MobileFace": 0.301611,
}
