import logging
from psg_utils.dataset.sleep_study_dataset import H5Dataset
from psg_utils.dataset.queue import StudyLoader, LimitationQueue, LazyQueue, EagerQueue

logger = logging.getLogger(__name__)


QUEUE_TYPE_TO_CLS = {
    "limitation": LimitationQueue,
    "lazy": LazyQueue,
    'eager': EagerQueue
}


def get_data_queues(datasets,
                    queue_type,
                    max_loaded_per_dataset,
                    num_access_before_reload,
                    n_load_processes=7,
                    study_loader=None):
    """
    TODO.

    Args:

    Returns:

    """
    map_ = {'eager': EagerQueue,
            'lazy': LazyQueue,
            'limitation': LimitationQueue}
    queue_type = map_[queue_type.lower()]
    logger.info("Using data queue type: {}".format(queue_type.__name__))

    if queue_type is LimitationQueue and study_loader is None:
        # Get loader for limitation queue(s)
        max_loaded = (max_loaded_per_dataset or 0) * len(datasets)
        study_loader = StudyLoader(n_load_processes=n_load_processes,
                                   max_queue_size=max_loaded or None)
    else:
        study_loader = None

    dataset_queues = []
    for dataset in datasets:
        if max_loaded_per_dataset >= len(dataset) and queue_type is LimitationQueue:
            # TODO: Implement load/access_time_random_channel_selector for EagerQueue, see NotImplementedError below.
            logger.warning(f"Using '{queue_type.__name__}' for dataset {dataset} even though max_loaded_per_dataset = {max_loaded_per_dataset} "
                           f">= len(dataset) = {len(dataset)})")
            # queue_type = EagerQueue
        if queue_type is EagerQueue and not isinstance(dataset, H5Dataset) and \
                (any([getattr(ss, 'load_time_random_channel_selector', False) or
                      getattr(ss, 'access_time_random_channel_selector', False) for ss in dataset])):
            raise NotImplementedError(
                "The 'eager' data loading queue currently does not support datasets with "
                "the 'channel_sampling_groups' attribute set. "
                "If you want to train using random channel combinations, either "
                "pre-process the data using the 'ut preprocess' command and then re-run "
                "training using 'ut train --preprocessed', or run training with the "
                "limitation queue loader using the '--train_queue_type "
                "limitation' command."
            )
        dataset_queues.append(queue_type(
            dataset=dataset,
            max_loaded=max_loaded_per_dataset,
            num_access_before_reload=num_access_before_reload,  # TODO
            preload_now=True,
            await_preload=False,
            study_loader=study_loader
        ))
    if study_loader:
        study_loader.join()
    return dataset_queues, study_loader
