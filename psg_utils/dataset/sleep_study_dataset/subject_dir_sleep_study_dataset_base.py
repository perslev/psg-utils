import logging
import os
import numpy as np
from abc import ABC
from typing import Union
from psg_utils.errors import CouldNotLoadError
from psg_utils.dataset.utils import find_subject_folders
from psg_utils.dataset import SleepStudy
from psg_utils.dataset.sleep_study_dataset.abc_sleep_study_dataset \
    import AbstractBaseSleepStudyDataset
from psg_utils.time_utils import TimeUnit

logger = logging.getLogger(__name__)


class SubjectDirSleepStudyDatasetBase(AbstractBaseSleepStudyDataset, ABC):
    """
    Represents a collection of SleepStudy objects
    """
    def __init__(self,
                 data_dir: str,
                 sleep_study_class: Union[type(SleepStudy)] = SleepStudy,
                 folder_regex: str = r'^(?!views).*$',
                 psg_regex: str = None,
                 hyp_regex: str = None,
                 no_labels: bool = False,
                 period_length: [int, float] = 30,
                 time_unit: Union[TimeUnit, str] = TimeUnit.SECOND,
                 internal_time_unit: Union[TimeUnit, str] = TimeUnit.MILLISECOND,
                 on_overlapping: str = "RAISE",
                 annotation_dict: dict = None,
                 identifier: str = None,
                 no_log: bool = False):
        """
        Initialize a SleepStudyDataset from a directory storing one or more
        sub-directories each corresponding to a sleep/PSG study.
        Each sub-dir will be represented by a SleepStudy object.

        Args:
            data_dir:                (string)     Path to the data directory
            sleep_study_class        (class)      TODO
            folder_regex:            (string)     Regex that matches folders to
                                                  consider within the data_dir.
            psg_regex:               (string)     Regex that matches files to
                                                  consider 'PSG' (data) within each
                                                  subject folder.
                                                  Passed to each SleepStudy.
            hyp_regex:               (string)     As psg_regex, but for hypnogram/
                                                  sleep stages/label files.
                                                  Passed to each SleepStudy.
            no_labels:               (bool)       TODO
            period_length            (int/float)  Sleep 'epoch' (segment/period) length in units 'time_unit' (see below)
            time_unit                (TimeUnit)   TimeUnit object specifying the unit of time of 'period_length'
            internal_time_unit       (TimeUnit)   TimeUnit object specifying the unit of time to use internally for storing
                                                  times. Affects the values returned by methods or attributes such as
                                                  self.period_length.
            on_overlapping:          (str)        One of 'FIRST', 'LAST', 'MAJORITY', , 'RAISE'.
                                                  Controls the behaviour when a discrete period of length
                                                  self.period_length overlaps 2 or more different classes
                                                  in the original hypnogram. See SparseHypnogram.get_period_at_time for
                                                  details.
            annotation_dict:         (dict)       Dictionary mapping labels as storred in the hyp files to
                                                  label integer values.
            identifier:              (string)     Dataset ID/name
            no_log:                  (bool)       Do not log dataset details on init
        """
        if not no_labels and bool(psg_regex) != bool(hyp_regex):
            raise RuntimeError("Must specify both or none of the 'psg_regex' "
                               "and 'hyp_regex' arguments.")
        self.data_dir = os.path.abspath(data_dir)
        # Init base class
        super(SubjectDirSleepStudyDatasetBase, self).__init__(
            identifier=identifier or os.path.split(self.data_dir)[-1],
            no_log=True
        )
        # Get list of subject folders in the data_dir according to folder_regex
        subject_folders = find_subject_folders(self.data_dir, folder_regex)
        if len(subject_folders) == 0:
            raise RuntimeError("Found no subject folders in data directory "
                               "{} using folder regex {}.".format(self.data_dir,
                                                                  folder_regex))
        # Initialize SleepStudy objects
        pairs = []
        for subject_dir in subject_folders:
            ss = sleep_study_class(
                subject_dir=subject_dir,
                psg_regex=psg_regex,
                hyp_regex=hyp_regex,
                no_hypnogram=no_labels,
                period_length=period_length,
                time_unit=time_unit,
                internal_time_unit=internal_time_unit,
                on_overlapping=on_overlapping,
                annotation_dict=annotation_dict,
                load=False
            )
            pairs.append(ss)
        self.add_pairs(pairs)
        if not no_log:
            self.log()

    def load(self, N=None, random_order=True):
        """
        Load all or a subset of stored SleepStudy objects
        Data is loaded using a thread pool with one thread per SleepStudy.

        Args:
            N:              Number of SleepStudy objects to load. Defaults to
                            loading all.
            random_order:   Randomly select which of the stored objects to load
                            rather than starting from the beginning. Only has
                            an effect with N != None
        Returns:
            self, reference to the SleepStudyDataset object
        """
        from concurrent.futures import ThreadPoolExecutor
        if N is None:
            N = len(self)
            random_order = False
        not_loaded = self.non_loaded_pairs
        if random_order:
            to_load = np.random.choice(not_loaded, size=N, replace=False)
        else:
            to_load = not_loaded[:N]
        self.log("Loading {}/{} SleepStudy objects...".format(len(to_load),
                                                              len(self)))
        pool = ThreadPoolExecutor(max_workers=min(len(to_load), 7))
        res = pool.map(lambda x: x.load(), to_load)
        try:
            for i, ss in enumerate(res):
                print(" -- {}/{}".format(i+1, len(to_load)), end="\r", flush=True)
        except CouldNotLoadError as e:
            raise CouldNotLoadError("Could not load sleep study {}."
                                    " Please refer to the above "
                                    "traceback.".format(e.study_id)) from e
        finally:
            pool.shutdown()
        return self
