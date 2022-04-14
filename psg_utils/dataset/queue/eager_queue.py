import logging
from psg_utils.dataset.queue.base_queue import BaseQueue
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class EagerQueue(BaseQueue):
    """
    Implements a queue-like object (same API interface as LoadQueue), but one
    that loads all data immediately when initialized.
    This is useful for wrapping a smaller collection of data in an object that
    behaves similar to the training queue object, but where all data is loaded
    up-front.
    """
    def __init__(self, dataset, **kwargs):
        """
        TODO
        Args:
            dataset:
        """
        super(EagerQueue, self).__init__(
            dataset=dataset
        )
        self.dataset.load()

    @staticmethod
    def check_loaded(study):
        if not study.loaded:
            raise RuntimeError("Some process unloaded sleep study '{}'; this "
                               "is unexpected behaviour when using the "
                               "EagerQueue object, which expects all data to "
                               "be loaded at all times")
        return study

    def __iter__(self):
        for i in range(len(self.dataset.pairs)):
            with self.get_study_by_idx(i) as ss:
                yield ss

    @contextmanager
    def get_random_study(self):
        yield self.check_loaded(super().get_random_study())

    @contextmanager
    def get_study_by_idx(self, study_idx):
        yield self.check_loaded(super().get_study_by_idx(study_idx))

    @contextmanager
    def get_study_by_id(self, study_id):
        yield self.check_loaded(super().get_study_by_id(study_id))
