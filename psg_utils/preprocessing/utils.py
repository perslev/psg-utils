import logging

logger = logging.getLogger(__name__)


def set_preprocessing_pipeline(*datasets, hparams):
    """
    Helper function which calls the following methods on a SleepStudyDataset
    like object with parameters inferred from a YAMLHparams object:
    - SleepStudyDataset.set_select_channels()
    - SleepStudyDataset.set_alternative_select_channels()
    - [if applicable] SleepStudyDataset.set_sample_rate()
    - [if applicable] SleepStudyDataset.set_strip_func()
    - [if applicable] SleepStudyDataset.set_quality_control_func()
    - [if applicable] SleepStudyDataset.set_scaler()
    - [if applicable] SleepStudyDataset.set_channel_sampling_groups()
    - [if applicable] SleepStudyDataset.set_filtering_settings()

    Args:
        *datasets: Any number of SleepStudyDataset-like objects
        hparams:   A YAMLHparams object parameterised the 3 methods called.
    """
    # Select channels if specified
    select = hparams.get("select_channels", [])
    list(map(lambda ds: ds.set_select_channels(select), datasets))

    # Set alternative select channels if specified
    alt_select = hparams.get("alternative_select_channels", [])
    list(map(lambda ds: ds.set_alternative_select_channels(alt_select), datasets))

    # Set load/access time channel sampler if specified
    channel_groups = tuple(hparams.get("channel_sampling_groups", []))
    dataset_types = list(map(type, datasets))
    if channel_groups:
        try:
            list(map(lambda ds: ds.set_channel_sampling_groups(*channel_groups), datasets))
        except AttributeError as e:
            raise ValueError(f"One or more of the dataset types in {dataset_types} do not support "
                             f"setting the 'set_channel_sampling_groups' attribute.") from e

    # Set sample rate
    if hasattr(datasets[0], 'set_sample_rate'):
        sample_rate = hparams.get("set_sample_rate", None)
        list(map(lambda ds: ds.set_sample_rate(sample_rate), datasets))

    # Apply strip function if specified
    strip_settings = hparams.get("strip_func")
    if strip_settings and hasattr(datasets[0], 'set_strip_func'):
        list(map(lambda ds: ds.set_strip_func(**strip_settings), datasets))

    # Set filtering
    filter_settings = hparams.get("filter_settings")
    if filter_settings and hasattr(datasets[0], 'set_filter_settings'):
        list(map(lambda ds: ds.set_filter_settings(**filter_settings), datasets))

    # Set notch filtering
    notch_filter_settings = hparams.get("notch_filter_settings")
    if notch_filter_settings and hasattr(datasets[0], 'set_notch_filter_settings'):
        list(map(lambda ds: ds.set_notch_filter_settings(**notch_filter_settings), datasets))

    # Apply quality control function if specified
    quality_settings = hparams.get("quality_control_func")
    if quality_settings and hasattr(datasets[0], 'set_quality_control_func'):
        list(map(lambda ds: ds.set_quality_control_func(**quality_settings), datasets))

    # Set misc attribute on dataset if specified
    misc = hparams.get('misc', {})
    if misc and hasattr(datasets[0], 'set_misc_dict'):
        list(map(lambda ds: ds.set_misc_dict(misc), datasets))

    # Set scaler
    if hasattr(datasets[0], 'set_scaler'):
        scl = hparams.get("scaler") or "RobustScaler"
        list(map(lambda ds: ds.set_scaler(scl), datasets))
