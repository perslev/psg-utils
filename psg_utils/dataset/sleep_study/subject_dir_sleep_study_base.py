"""
Implements the SleepStudyBase class which represents a sleep study (PSG)
"""

import logging
import os
import numpy as np
from abc import ABC
from typing import Tuple, Union
from psg_utils import Defaults
from psg_utils.dataset.utils import find_psg_and_hyp
from psg_utils.dataset.sleep_study.abc_sleep_study import AbstractBaseSleepStudy
from psg_utils.time_utils import TimeUnit

logger = logging.getLogger(__name__)


class SubjectDirSleepStudyBase(AbstractBaseSleepStudy, ABC):
    def __init__(self,
                 subject_dir,
                 psg_regex=None,
                 hyp_regex=None,
                 header_regex=None,
                 no_hypnogram=None,
                 annotation_dict=None,
                 period_length: [int, float] = 30,
                 time_unit: Union[TimeUnit, str] = TimeUnit.SECOND,
                 internal_time_unit: Union[TimeUnit, str] = TimeUnit.MILLISECOND,
                 on_overlapping: str = "RAISE"):
        """
        Initialize a SubjectDirSleepStudyBase object from PSG/HYP data

        PSG: A file that stores polysomnography (PSG) data
        HYP: A file that stores the sleep stages / annotations for the PSG

        Takes a path pointing to a directory in which two or more files are
        located. One of those files should be a PSG (data) file and unless
        no_hypnogram == True another should be a hypnogram/sleep stages/labels
        file. The PSG(/HYP) files are automatically automatically inferred
        using a set of simple rules when psg_regex or hyp_regex are None
        (refer to the 'psg_utils.dataset.utils.find_psg_and_hyp' function).
        Otherwise, the psg_regex and/or hyp_regex is used to match against
        folder content. Each regex should have exactly one match within
        'subject_dir'.

        Args:
            subject_dir:      (str)        File path to a directory storing the
                                             subject data.
            psg_regex:        (str)        Optional regex used to select PSG file
            hyp_regex:        (str)        Optional regex used to select HYP file
            header_regex:     (str)        Optional regex used to select a header file
                                           OBS: Rarely used as most formats store headers internally, or
                                             have header paths which are inferrable from the psg_path.
            no_hypnogram      (bool)       Initialize without ground truth data.
            annotation_dict   (dict)       A dictionary mapping from labels in the
                                           hyp_file_path file to integers
           period_length      (int/float)  Sleep 'epoch' (segment/period) length in units 'time_unit' (see below)
           time_unit          (TimeUnit)   TimeUnit object specifying the unit of time of 'period_length'
           internal_time_unit (TimeUnit)   TimeUnit object specifying the unit of time to use internally for storing
                                           times. Affects the values returned by methods or attributes such as
                                           self.period_length.
           on_overlapping:    (str)        One of 'FIRST', 'LAST', 'MAJORITY', 'RAISE'. Controls the behaviour when a discrete
                                             period of length self.period_length overlaps 2 or more different classes
                                             in the original hypnogram. See SparseHypnogram.get_period_at_time for
                                             details.
        """
        super(SubjectDirSleepStudyBase, self).__init__(
            annotation_dict=annotation_dict,
            no_hypnogram=no_hypnogram,
            period_length=period_length,
            time_unit=time_unit,
            internal_time_unit=internal_time_unit,
            on_overlapping=on_overlapping
        )
        self.subject_dir = os.path.abspath(subject_dir)
        try:
            psg, hyp, header = find_psg_and_hyp(subject_dir=self.subject_dir,
                                                psg_regex=psg_regex,
                                                hyp_regex=hyp_regex,
                                                header_regex=header_regex,
                                                no_hypnogram=no_hypnogram)
        except (ValueError, RuntimeError) as e:
            raise ValueError("Could not uniquely infer PSG/HYP files in subject"
                             " directory {}. Consider specifying/correcting "
                             "one or both of the psg_regex and hyp_regex "
                             "parameters to explicitly select the appropriate "
                             "file(s) within the "
                             "subject dir.".format(repr(subject_dir))) from e
        self.psg_file_path = psg
        self.hyp_file_path = hyp if not no_hypnogram else None
        self.header_file_path = header  # OBS: Most often None

    @property
    def identifier(self) -> str:
        """
        Returns an ID, which is simply the name of the directory storing
        the data
        """
        return os.path.split(self.subject_dir)[-1]

    @property
    def n_classes(self) -> int:
        """ Returns the number of classes represented in the hypnogram """
        return self.hypnogram.n_classes

    def get_psg_periods_by_idx(self, start_idx: int, n_periods: int = 1, channel_indices: list = None) -> np.ndarray:
        """
        Returns periods from the PSG in shape [n_periods, self.data_per_period, n_channels].

        Args:
            start_idx (int):        Index of first period to return
            n_periods (int):        The number of periods to return
            channel_indices (list): Optional list of channel indices to extract from PSG. Extracts all with None.

        Returns:
            psg: ndarray of shape [n_periods, self.data_per_period, n_channels]
        """
        self._assert_period_index_bounds(start_idx + n_periods - 1)
        data_start_idx = start_idx * self.data_per_period
        data_end_idx = data_start_idx + (self.data_per_period * n_periods)
        psg = self.psg[data_start_idx:data_end_idx]
        if channel_indices is not None:
            psg = psg[:, channel_indices]
        return psg.reshape([n_periods, self.data_per_period, psg.shape[-1]])

    def get_hyp_periods_by_idx(self, start_idx: int, n_periods: int = 1, on_overlapping: Union[str, None] = None) -> np.ndarray:
        """
        Returns periods from the hypnogram in shape [n_periods].

        Args:
            start_idx (int):              Index of first period to return
            n_periods (int):              The number of periods to return
            on_overlapping (str or None): If str one of 'FIRST', 'LAST', 'MAJORITY'. Controls the behaviour when a
                                          discrete period of length self.period_length overlaps 2 or more different
                                          classes in the original hypnogram. See SparseHypnogram.get_period_at_time
                                          for details. Default with on_overlapping = None is self.on_overlapping.

        Returns:
            hyp: ndarray of shape [n_periods]
        """
        self._assert_period_index_bounds(start_idx + n_periods - 1)
        hyp = np.empty(shape=[n_periods], dtype=Defaults.HYP_DTYPE)
        for i, idx in enumerate(range(start_idx, start_idx+n_periods)):
            period_start_time = self.period_idx_to_time(idx)
            hyp[i] = self.hypnogram.get_period_at_time(
                time=period_start_time,
                time_unit=self.time_unit,
                on_overlapping=on_overlapping or self.on_overlapping
            )
        return hyp

    def get_psg_as_array(self):
        """
        Returns the PSG stored in self.psg as a single, flat ndarray of shape [-1, n_channels]
        """
        return self.psg  # Already an ndarray of the correct shape
