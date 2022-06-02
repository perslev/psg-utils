import logging
from psg_utils.dataset.queue.base_queue import BaseQueue
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class LazyQueue(BaseQueue):
    """
    Implements a queue-like object (same API interface as LoadQueue), but one
    that only loads data just-in-time when requested.
    This is useful for wrapping e.g. validation data in an object that behaves
    similar to the training queue object, but without consuming memory before
    needing to do validation.
    """
    def __init__(self, dataset, **kwargs):
        """
        TODO
        Args:
            dataset:
        """
        super(LazyQueue, self).__init__(
            dataset=dataset
        )

    @contextmanager
    def get_random_study(self):
        study = super().get_random_study()
        with study.loaded_in_context():
            yield study

    @contextmanager
    def get_study_by_idx(self, study_idx):
        study = super().get_study_by_idx(study_idx)
        with study.loaded_in_context():
            yield study

    @contextmanager
    def get_study_by_id(self, study_id):
        study = super().get_study_by_id(study_id)
        with study.loaded_in_context():
            yield study
