import logging
import numpy as np
from numpy.lib.stride_tricks import as_strided
from psg_utils import Defaults

logger = logging.getLogger(__name__)


# A list of default triplet replacement rules for hypnogram smoothing
# As per
TRIPLET_RULES = [
    (["W", "REM", "N2"], ["W", "N1", "N2"]),
    (["N1", "REM", "N2"], ["N1", "N1", "N2"]),
    (["N2", "N1", "N2"], ["N2", "N2", "N2"]),
    (["N2", "N3", "N2"], ["N2", "N2", "N2"]),
    (["N2", "REM", "N2"], ["N2", "N2", "N2"]),
    (["N3", "N2", "N3"], ["N3", "N3", "N3"]),
    (["REM", "W", "REM"], ["REM", "REM", "REM"]),
    (["REM", "N1", "REM"], ["REM", "REM", "REM"]),
    (["REM", "N2", "REM"], ["REM", "REM", "REM"])
]


def get_translated_triplet_rules(translation_map=None):
    """ TODO """
    translation_map = translation_map or Defaults.get_stage_string_to_class_int()
    translation_map = np.vectorize(translation_map.get)
    new_map = []
    for s1, s2 in TRIPLET_RULES:
        s1 = translation_map(s1)
        s2 = translation_map(s2)
        new_map.append((s1, s2))
    return new_map


def find_matches(scores, pattern):
    """ TODO """
    strided = as_strided(scores, shape=(len(scores) - len(pattern) + 1, len(pattern)),
                         strides=(scores.strides[0],)*2)
    return np.where(np.all(strided == pattern, axis=1))[0]


def apply_substitution_rules(scores, substitution_rules, verbose=False):
    """ TODO """
    scores = scores.copy()
    for pattern, target in substitution_rules:
        inds = find_matches(scores, pattern)
        for ind in inds:
            scores[ind:ind+len(pattern)] = target
        if verbose:
            print(f"Applied rule {pattern}-->{target} at {len(inds)} indices")
    return scores


def replace_before_with(scores, before_stage, replace, to, stage_string_to_class_int_map=None):
    """ TODO """
    scores = scores.copy()
    if isinstance(before_stage, str):
        stage_map = stage_string_to_class_int_map or Defaults.get_stage_string_to_class_int()
        before_stage, replace, to = map(lambda s: stage_map[s], (before_stage, replace, to))
    first_appearence = np.where(scores == before_stage)[0]
    if len(first_appearence):
        scores_considered = scores[:first_appearence[0]]
        scores_considered = np.where(scores_considered == replace, to, scores_considered)
        scores[:first_appearence[0]] = scores_considered
    return scores
