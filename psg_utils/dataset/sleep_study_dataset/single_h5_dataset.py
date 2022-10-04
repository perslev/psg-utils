import logging
import os
import re
import h5py
import atexit
from typing import Union
from psg_utils.io.channels import RandomChannelSelector
from psg_utils.dataset.sleep_study_dataset.abc_sleep_study_dataset import AbstractBaseSleepStudyDataset
from psg_utils.dataset.sleep_study import H5SleepStudy
from psg_utils.time_utils import TimeUnit

logger = logging.getLogger(__name__)


class H5Dataset(AbstractBaseSleepStudyDataset):
    def __init__(self,
                 h5_dataset_obj,
                 identifier=None,
                 annotation_dict=None,
                 period_length: [int, float] = 30,
                 time_unit: Union[TimeUnit, str] = TimeUnit.SECOND,
                 internal_time_unit: Union[TimeUnit, str] = TimeUnit.MILLISECOND,
                 no_log=False):
        self.h5_dataset_obj = h5_dataset_obj
        super(H5Dataset, self).__init__(
            identifier=identifier or self.h5_dataset_obj.name.lstrip("/"),
            no_log=True
        )
        pairs = []
        for pair_id in self.h5_dataset_obj:
            try:
                pair = H5SleepStudy(
                    self.h5_dataset_obj[pair_id],
                    annotation_dict=annotation_dict,
                    period_length=period_length,
                    time_unit=time_unit,
                    internal_time_unit=internal_time_unit
                )
            except KeyError:
                continue  # missing data, TODO, temp
            else:
                pairs.append(pair)
        self.add_pairs(pairs)
        if not no_log:
            self.log()

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "H5Dataset(identifier={}, members={}, loaded={})".format(
            self.identifier, len(self.h5_dataset_obj), len(self.pairs)
        )

    def load(self, n=None, random_order=None):
        """ No effect """
        if not bool(self.h5_dataset_obj):
            raise RuntimeError("HDF5 archive was closed unexpectedly.")

    def set_channel_sampling_groups(self, *channel_groups):
        """
        TODO

        Args:
            channel_groups:
        """
        if len(channel_groups) == 0 or channel_groups[0] is None:
            random_selector = None
        else:
            random_selector = RandomChannelSelector(*channel_groups)
        self.log("Setting access-time random channel selector: "
                 "{}".format(random_selector))
        for ss in self:
            ss.access_time_random_channel_selector = random_selector


class SingleH5Dataset:
    """
    TODO
    """
    def __init__(self,
                 h5_path,
                 identifier=None):
        """
        Initialize a dataset from a single HDF5 file storing all (preprocessed)
        data. Usually used in conjungtion with the output of 'ut preprocess'.

        Args:
            TODO
        """
        self.h5_path = os.path.abspath(h5_path)
        self.identifier = identifier or \
                          os.path.splitext(os.path.split(h5_path)[-1])[0]
        if not h5py.is_hdf5(self.h5_path):
            raise OSError("Invalid file at path {}. Is not a HDF5 file."
                          "".format(self.h5_path))

        # Open HDF5 archive, register close func on exit
        self.h5_object = h5py.File(self.h5_path, "r")
        atexit.register(lambda: self.h5_object.close())
        logger.info(str(self))

    def __str__(self):
        return "SingleH5Dataset(identifier={}, path={})".format(
            self.identifier, self.h5_path)

    def __repr__(self):
        return str(self)

    def close(self):
        self.h5_object.close()

    def get_datasets(self,
                     load_match_regex=None,
                     period_length: [int, float] = 30,
                     time_unit: Union[TimeUnit, str] = TimeUnit.SECOND,
                     internal_time_unit: Union[TimeUnit, str] = TimeUnit.MILLISECOND,
                     annotation_dict=None,
                     no_log=False):
        regex = re.compile(load_match_regex or ".*")
        dataset_names = self.h5_object.keys()
        datasets = []
        for name in dataset_names:
            for dataset in self.h5_object[name]:
                dataset_h5 = self.h5_object[name][dataset]
                if not re.match(regex, dataset_h5.name):
                    continue
                datasets.append(H5Dataset(
                    h5_dataset_obj=dataset_h5,
                    identifier=dataset_h5.name.lstrip("/"),
                    annotation_dict=annotation_dict,
                    period_length=period_length,
                    time_unit=time_unit,
                    internal_time_unit=internal_time_unit,
                    no_log=no_log
                ))
        return datasets
