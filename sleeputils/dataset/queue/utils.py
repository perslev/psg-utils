import logging
from sleeputils.utils import ensure_list_or_tuple
from sleeputils.dataset.queue import (StudyLoader, LimitationQueue,
                                      LazyQueue, EagerQueue)

logger = logging.getLogger(__name__)


QUEUE_TYPE_TO_CLS = {
    "limitation": LimitationQueue,
    "lazy": LazyQueue,
    'eager': EagerQueue
}


def get_dataset_queues(datasets,
                       queue_type,
                       n_load_threads=7,
                       **kwargs):
    """
    TODO

    :param datasets:
    :param queue_type:
    :param n_load_threads:
    :param kwargs:
    :return:
    """
    if datasets is None:
        return None
    datasets = ensure_list_or_tuple(datasets)

    # Prepare study loader object
    max_loaded = kwargs.get("max_loaded_per_dataset", 0) * len(datasets)
    study_loader = StudyLoader(n_threads=n_load_threads,
                               max_queue_size=max_loaded or None)

    # Get a queue for each dataset
    queues = []
    queue_cls = QUEUE_TYPE_TO_CLS[queue_type.lower()]
    for dataset in datasets:
        queue = queue_cls(
            dataset=dataset,
            max_loaded=kwargs.get("max_loaded_per_dataset"),
            study_loader=study_loader,
            **kwargs
        )
        queues.append(queue)
    return queues


def assert_all_loaded(pairs, raise_=True):
    """
    Returns True if all SleepStudy objects in 'pairs' have the 'loaded'
    property set to True, otherwise returns False.

    If raise_ is True, raises a NotImplementedError if one or more objects are
    not loaded. Otherwise, returns the value of the assessment.

    Temp. until queue functionality implemented
    """
    loaded_pairs = [p for p in pairs if p.loaded]
    if len(loaded_pairs) != len(pairs):
        if raise_:
            raise NotImplementedError("BatchSequence currently requires all"
                                      " samples to be loaded")
        else:
            return False
    return True
