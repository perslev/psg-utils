import logging
import numpy as np

logger = logging.getLogger(__name__)


class _Defaults:
    """
    Stores and potentially updates default values for sleep stages etc.
    """
    # Standardized string representation for 5 typical sleep stages
    AWAKE = ["W", 0]
    NON_REM_STAGE_1 = ["N1", 1]
    NON_REM_STAGE_2 = ["N2", 2]
    NON_REM_STAGE_3 = ["N3", 3]
    REM = ["REM", 4]
    UNKNOWN = ["UNKNOWN", 5]

    # Visualization defaults
    STAGE_COLORS = ["darkblue", "darkred",
                    "darkgreen", "darkcyan",
                    "darkorange", "black"]

    # Default segmentation length in seconds
    PERIOD_LENGTH_SEC = 30

    # Default dtypes
    PSG_DTYPE = np.float32
    HYP_DTYPE = np.uint8

    @classmethod
    def get_vectorized_stage_colors(cls):
        import numpy as np
        map_ = {i: col for i, col in enumerate(cls.STAGE_COLORS)}
        return np.vectorize(map_.get)

    @classmethod
    def get_stage_lists(cls, include_unkown_class=True):
        classes = [cls.AWAKE, cls.NON_REM_STAGE_1, cls.NON_REM_STAGE_2,
                   cls.NON_REM_STAGE_3, cls.REM, cls.UNKNOWN]
        if include_unkown_class:
            return classes
        else:
            return classes[:-1]

    @classmethod
    def get_stage_string_to_class_int(cls):
        # Dictionary mapping from the standardized string rep to integer
        # representation
        return {s[0]: s[1] for s in cls.get_stage_lists()}

    @classmethod
    def get_class_int_to_stage_string(cls, include_unkown_class=True):
        # Dictionary mapping from integer representation to standardized
        # string rep
        return {s[1]: s[0] for s in cls.get_stage_lists(include_unkown_class)}

    @classmethod
    def get_default_period_length(cls):
        logger.warning("Using default period length of {} seconds.".format(cls.PERIOD_LENGTH_SEC))
        return cls.PERIOD_LENGTH_SEC
