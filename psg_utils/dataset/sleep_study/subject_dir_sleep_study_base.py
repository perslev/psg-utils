"""
Implements the SleepStudyBase class which represents a sleep study (PSG)
"""

import logging
import os
import numpy as np
from typing import Tuple, Union
from psg_utils import Defaults
from psg_utils.dataset.utils import find_psg_and_hyp
from psg_utils.dataset.sleep_study.abc_sleep_study import AbstractBaseSleepStudy
from psg_utils.time_utils import TimeUnit

logger = logging.getLogger(__name__)


class SubjectDirSleepStudyBase(AbstractBaseSleepStudy):
    def __init__(self,
                 subject_dir,
                 psg_regex=None,
                 hyp_regex=None,
                 header_regex=None,
                 period_length_sec=None,
                 no_hypnogram=None,
                 annotation_dict=None):
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
            subject_dir:      (str)    File path to a directory storing the
                                       subject data.
            psg_regex:        (str)    Optional regex used to select PSG file
            hyp_regex:        (str)    Optional regex used to select HYP file
            header_regex:     (str)    Optional regex used to select a header file
                                       OBS: Rarely used as most formats store headers internally, or
                                            have header paths which are inferrable from the psg_path.
            period_length_sec (int)    Sleep 'epoch' (segment/period) length in
                                       seconds
            no_hypnogram      (bool)   Initialize without ground truth data.
            annotation_dict   (dict)   A dictionary mapping from labels in the
                                       hyp_file_path file to integers
        """
        super(SubjectDirSleepStudyBase, self).__init__(
            annotation_dict=annotation_dict,
            period_length_sec=period_length_sec,
            no_hypnogram=no_hypnogram
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

    @property
    def recording_length_sec(self) -> float:
        """ Returns the total length (in seconds) of the PSG recording """
        return self.get_psg_shape()[0] / self.sample_rate

    def get_full_hypnogram(self, on_overlapping: str = "RAISE") -> np.ndarray:
        """
        Returns the full (dense) hypnogram

        Args:
            on_overlapping: str, One of 'FIRST', 'LAST', 'MAJORITY'. Controls the behaviour when a descrete
                                 period of length self.period_length overlaps 2 or more different classes in the
                                 original hypnogram. See SparseHypnogram.get_period_at_time for details.

        Returns:
            An ndarray of shape [self.n_periods, 1] of class labels
        """
        dense_hypnogram = self.hypnogram.to_dense(on_overlapping)
        return dense_hypnogram["sleep_stage"].to_numpy().reshape(-1, 1)

    def get_periods_by_idx(self, start_idx: int, end_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a range of period of {X, y} data by indices
        Period starting at second 0 is index 0.

        Returns [N periods = end_idx - start_idx + 1] periods

        Args:
            start_idx (int): Index of first period to return
            end_idx   (int): Index of last period to return (inclusive)

        Returns:
            X: ndarray of shape [N periods, self.data_per_period, C]
            y: ndarray of shape [N periods, 1]
        """
        indices = list(range(start_idx, end_idx+1))
        x = np.empty(shape=[len(indices), self.data_per_period, len(self.select_channels)],
                     dtype=Defaults.PSG_DTYPE)
        y = np.empty(shape=[len(indices), 1], dtype=Defaults.HYP_DTYPE)
        for i, idx in enumerate(indices):
            x_period, y_period = self.get_period_by_idx(idx)
            x[i] = x_period
            y[i] = y_period
        return x, y

    def get_psg_period_at_sec(self, second: [int, float]) -> np.ndarray:
        """
        Get PSG period starting at second 'second'.

        Returns:
            X: An ndarray of shape [self.data_per_period, self.n_channels]
        """
        if second % self.period_length_sec:
            raise ValueError("Invalid second of {}, not divisible by period "
                             "length of {} "
                             "seconds".format(second, self.period_length_sec))
        return self.extract_from_psg(start_second=second,
                                     end_second=second+self.period_length_sec)

    def get_stage_at_sec(self, second: [int, float]) -> Union[int, np.ndarray]:
        """
        TODO

        Args:
            second:

        Returns:

        """
        return self.hypnogram.get_stage_at_time(second, TimeUnit.SECOND)

    def get_all_periods(self, on_overlapping: str = "RAISE") -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Returns the full (dense) data of the SleepStudy

        Args:
        on_overlapping: str, One of 'FIRST', 'LAST', 'MAJORITY'. Controls the behaviour when a descrete
                             period of length self.period_length overlaps 2 or more different classes in the
                             original hypnogram. See SparseHypnogram.get_period_at_time for details.

        Returns:
            X: An ndarray of shape [self.n_periods,
                                    self.data_per_period,
                                    self.n_channels]
            y: An ndarray of shape [self.n_periods, 1] (if self.no_hypnogram == False)
        """
        X = self.get_full_psg().reshape(shape=[-1, self.data_per_period, self.n_channels])
        if self.no_hypnogram:
            return X
        y = self.get_full_hypnogram(on_overlapping)
        if len(X) != len(y):
            err_msg = ("Length of PSG array does not match length dense "
                       "hypnogram array ({} != {}). If hypnogram "
                       "is longer, consider if a trailing or leading "
                       "sleep stage should be removed. (you may use "
                       "SleepStudyDataset.set_hyp_strip_func())".format(len(X),
                                                                        len(y)))
            self.raise_err(ValueError, err_msg)
        return X, y
