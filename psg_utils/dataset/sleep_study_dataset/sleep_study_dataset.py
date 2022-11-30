import logging
from typing import Union
from psg_utils.io.channels import RandomChannelSelector
from psg_utils.dataset.sleep_study_dataset.subject_dir_sleep_study_dataset_base \
    import SubjectDirSleepStudyDatasetBase
from psg_utils.time_utils import TimeUnit

logger = logging.getLogger(__name__)


class SleepStudyDataset(SubjectDirSleepStudyDatasetBase):
    """
    Represents a collection of SleepStudy objects
    """
    def __init__(self,
                 data_dir,
                 folder_regex=r'^(?!views).*$',
                 psg_regex=None,
                 hyp_regex=None,
                 no_labels=False,
                 period_length: [int, float] = 30,
                 time_unit: Union[TimeUnit, str] = TimeUnit.SECOND,
                 internal_time_unit: Union[TimeUnit, str] = TimeUnit.MILLISECOND,
                 on_overlapping: str = "RAISE",
                 annotation_dict=None,
                 identifier=None,
                 no_log=False):
        """
        Initialize a SleepStudyDataset from a directory storing one or more
        sub-directories each corresponding to a sleep/PSG study.
        Each sub-dir will be represented by a SleepStudy object.

        Args:
            data_dir:                (string)     Path to the data directory
            folder_regex:            (string)     Regex that matches folders to
                                                  consider within the data_dir.
            psg_regex:               (string)     Regex that matches files to
                                                  consider 'PSG' (data) within each
                                                  subject folder.
                                                  Passed to each SleepStudy.
            hyp_regex:               (string)     As psg_regex, but for hypnogram/
                                                  sleep stages/label files.
                                                  Passed to each SleepStudy.
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
        super(SleepStudyDataset, self).__init__(
            data_dir=data_dir,
            folder_regex=folder_regex,
            psg_regex=psg_regex,
            hyp_regex=hyp_regex,
            no_labels=no_labels,
            period_length=period_length,
            time_unit=time_unit,
            internal_time_unit=internal_time_unit,
            on_overlapping=on_overlapping,
            annotation_dict=annotation_dict,
            identifier=identifier,
            no_log=no_log
        )

    def __str__(self):
        return "SleepStudyDataset(identifier: {}, N pairs: {}, N loaded: {})" \
               "".format(self.identifier, len(self), self.n_loaded)

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
        self.log("Setting load-time random channel selector: "
                 "{}".format(random_selector))
        for ss in self:
            ss.load_time_random_channel_selector = random_selector

    def set_scaler(self, scaler):
        """
        Sets the 'scaler' property on all stored SleepStudy objects.
        Please refer to the SleepStudy.scaler setter method
        """
        self.log("Setting '{}' scaler...".format(scaler))
        for ss in self:
            ss.scaler = scaler

    def set_sample_rate(self, sample_rate):
        """
        Sets the 'sample_rate' property on all stored SleepStudy objects.
        Please refer to the SleepStudy.sample_rate setter method
        """
        self.log("Setting sample rate of {} Hz".format(sample_rate))
        for ss in self:
            ss.sample_rate = sample_rate

    def set_strip_func(self, strip_func, **kwargs):
        """
        Sets the 'strip_func' property on all stored SleepStudy objects.
        Please refer to the SleepStudy.strip_func setter method
        """
        self.log("Setting '{}' strip function with parameters {}..."
                 "".format(strip_func, kwargs))
        for ss in self:
            ss.set_strip_func(strip_func, **kwargs)

    def set_filter_settings(self, **filter_settings):
        """
        Sets the 'filter_settings' property on all stored SleepStudy objects.
        Please refer to the SleepStudy.filter_settings setter method.
        """
        self.log(f"Setting filter settings: {filter_settings}...")
        for ss in self:
            ss.filter_settings = filter_settings

    def set_notch_filter_settings(self, **notch_filter_settings):
        """
        Sets the 'notch_filter_settings' property on all stored SleepStudy objects.
        Please refer to the SleepStudy.notch_filter_settings setter method.
        """
        self.log(f"Setting notch filter settings: {notch_filter_settings}...")
        for ss in self:
            ss.notch_filter_settings = notch_filter_settings

    def set_quality_control_func(self, quality_control_func, **kwargs):
        """
        Sets the 'quality_control_func' property on all stored SleepStudy
        objects. Please refer to the SleepStudy.quality_control_func setter
        method
        """
        self.log("Setting '{}' quality control function with "
                 "parameters {}...".format(quality_control_func, kwargs))
        for ss in self:
            ss.set_quality_control_func(quality_control_func, **kwargs)
