"""Layer catalogue: names, draw order and label-map colours.

Order is front (index 0) to back. Names follow the See-through / Live2D tag set
so outputs from both engines line up.
"""

# (name, label-map RGB colour)
PARTS = [
    ("front_hair",    (255, 64, 129)),
    ("side_hair_L",   (255, 128, 171)),
    ("side_hair_R",   (240, 98, 146)),
    ("eyebrow_L",     (121, 85, 72)),
    ("eyebrow_R",     (141, 110, 99)),
    ("eyelash_L",     (33, 33, 33)),
    ("eyelash_R",     (66, 66, 66)),
    ("iris_L",        (41, 121, 255)),
    ("iris_R",        (68, 138, 255)),
    ("eyewhite_L",    (224, 247, 250)),
    ("eyewhite_R",    (178, 235, 242)),
    ("mouth",         (213, 0, 0)),
    ("nose",          (255, 171, 145)),
    ("face",          (255, 224, 178)),
    ("headwear",      (255, 214, 0)),
    ("neckwear",      (0, 200, 83)),
    ("sleeve_L",      (124, 77, 255)),
    ("sleeve_R",      (149, 117, 205)),
    ("arm_L",         (255, 183, 77)),
    ("arm_R",         (255, 152, 0)),
    ("topwear",       (101, 31, 255)),
    ("neck",          (255, 204, 128)),
    ("bottomwear",    (0, 145, 234)),
    ("footwear_L",    (93, 64, 55)),
    ("footwear_R",    (78, 52, 46)),
    ("legwear_L",     (0, 184, 212)),
    ("legwear_R",     (0, 151, 167)),
    ("leg_L",         (255, 167, 38)),
    ("leg_R",         (251, 140, 0)),
    ("objects",       (174, 234, 0)),
    ("tail",          (100, 221, 23)),
    ("back_hair",     (194, 24, 91)),
]

NAMES = [p[0] for p in PARTS]
ID = {n: i + 1 for i, n in enumerate(NAMES)}          # 0 = background
COLOR = {n: c for n, c in PARTS}


def front_of(name):
    """Names drawn in front of `name`."""
    return NAMES[: NAMES.index(name)]
