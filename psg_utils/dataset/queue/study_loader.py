import logging
import os
from psg_utils.errors import CouldNotLoadError
from threading import Thread
from threading import Event as ThreadEvent
from multiprocessing import JoinableQueue, Process, Lock, Event, cpu_count
from time import sleep

logger = logging.getLogger(__name__)


def _load_func(load_queue, results_queue, load_errors_queue, lock, stop_event):
    """

    Args:
        load_queue:

    Returns:

    """
    while not stop_event.is_set():
        to_load, dataset_id = load_queue.get()
        try:
            to_load.load()
            results_queue.put((to_load, dataset_id))
        except CouldNotLoadError as e:
            with lock:
                logger.warning("[ERROR in StudyLoader] "
                               "Could not load study '{}' (error: {})".format(to_load, e))
            load_errors_queue.put((to_load, dataset_id))
        finally:
            load_queue.task_done()


def _gather_loaded(output_queue, registered_datasets, stop_event):
    while not stop_event.is_set():
        # Wait for studies in the output queue
        sleep_study, dataset_id = output_queue.get(block=True)
        load_put_function = registered_datasets[dataset_id][0]
        load_put_function(sleep_study)
        output_queue.task_done()


def _gather_errors(load_errors_queue, registered_datasets, stop_event):
    while not stop_event.is_set():
        # Wait for studies in the output queue
        sleep_study, dataset_id = load_errors_queue.get(block=True)
        error_put_function = registered_datasets[dataset_id][1]
        error_put_function(sleep_study)
        load_errors_queue.task_done()


def get_num_cpus(n_load_processes):
    """
    n_load_processes: [None, int] The number of processes to spin up for study loading.
                                  If None, uses int(os.environ['SLURM_JOB_CPUS_PER_NODE']) if set, otherwise
                                  multiprocessing.cpu_count() (using all visible CPUs).
    """
    if n_load_processes is None:
        slurm_cpus = os.environ.get('SLURM_JOB_CPUS_PER_NODE')
        if slurm_cpus:
            logger.info(f"Environment variable SLURM_JOB_CPUS_PER_NODE={slurm_cpus}")
            n_load_processes = int(slurm_cpus)
        else:
            num_cpus = cpu_count()
            logger.info(f"multiprocessing.cpu_count() returned N={num_cpus} visible CPUs.")
            n_load_processes = num_cpus
    return n_load_processes


class StudyLoader:
    """
    Implements a multithreading SleepStudy loading queue
    """
    def __init__(self,
                 n_load_processes=None,
                 max_queue_size=50):
        """
        Args:
            n_load_processes: [None, int] The number of processes to spin up for study loading.
                                          If None, uses int(os.environ['SLURM_JOB_CPUS_PER_NODE']) if set, otherwise
                                          multiprocessing.cpu_count() (using all visible CPUs).
        """
        # Setup load thread pool
        self.max_queue_size = max_queue_size
        self._load_queue = JoinableQueue(maxsize=self.max_queue_size)
        self._output_queue = JoinableQueue(maxsize=self.max_queue_size)
        self._load_errors_queue = JoinableQueue(maxsize=3)  # We probably want to raise
                                                            # an error if this queue
                                                            # gets to more than ~3!
        self.process_lock = Lock()

        # Init loading processes
        num_cpus = get_num_cpus(n_load_processes)
        logger.info(f"Creating StudyLoader with N={num_cpus} loading processes...")
        args = [self._load_queue, self._output_queue, self._load_errors_queue, self.process_lock]
        self.processes_and_threads = []
        self.stop_events = []
        for _ in range(num_cpus):
            stop_event = Event()
            p = Process(target=_load_func, args=args + [stop_event], daemon=True)
            p.start()
            self.processes_and_threads.append(p)
            self.stop_events.append(stop_event)

        # Prepare loaded studies gathering thread
        self._registered_datasets = {}
        gather_loaded_stop_event = ThreadEvent()
        self.gather_loaded_thread = Thread(target=_gather_loaded,
                                           args=(self._output_queue,
                                                 self._registered_datasets,
                                                 gather_loaded_stop_event),
                                           daemon=True)
        self.stop_events.append(gather_loaded_stop_event)
        self.processes_and_threads.append(self.gather_loaded_thread)
        self.gather_loaded_thread.start()

        # Start thread to collect load errors
        gather_errors_stop_event = ThreadEvent()
        self.gather_errors_thread = Thread(target=_gather_errors,
                                           args=(self._load_errors_queue,
                                                 self._registered_datasets,
                                                 gather_errors_stop_event),
                                           daemon=True)
        self.processes_and_threads.append(self.gather_errors_thread)
        self.stop_events.append(gather_errors_stop_event)
        self.gather_errors_thread.start()

    def stop(self):
        logger.info(f"Stopping N={len(self.processes_and_threads)} StudyLoader processes and threads...")
        for stop_event in self.stop_events:
            stop_event.set()
        for process_or_thread in self.processes_and_threads:
            process_or_thread.join()

    def qsize(self):
        """ Returns the qsize of the load queue """
        return self._load_queue.qsize

    @property
    def maxsize(self):
        return self.max_queue_size

    def join(self):
        """ Join on all queues """
        logger.info("Awaiting preload from {} (train) datasets".format(
            len(self._registered_datasets)
        ))
        self._load_queue.join()
        logger.info("Load queue joined...")
        self._output_queue.join()
        logger.info("Output queue joined...")
        self._load_errors_queue.join()
        logger.info("Errors queue joined...")

    def add_study_to_load_queue(self, study, dataset_id):
        if dataset_id not in self._registered_datasets:
            raise RuntimeError("Dataset {} is not registered. "
                               "Call StudyLoader.register_dataset before adding"
                               " items from that dataset to the loading "
                               "queue".format(dataset_id))
        if self.qsize() == self.maxsize:
            logger.warning("Loading queue seems about to block! "
                           "(max_size={}, current={}). "
                           "Sleeping until loading queue is empty "
                           "again.".format(self.maxsize,
                                           self.qsize()))
            while self.qsize() > 1:
                sleep(1)
        self._load_queue.put((study, dataset_id))

    def register_dataset(self, dataset_id, load_put_function, error_put_function):
        with self.process_lock:
            if dataset_id in self._registered_datasets:
                raise RuntimeWarning("A dataset of ID {} has already been "
                                     "registered.".format(dataset_id))
            self._registered_datasets[dataset_id] = (
                load_put_function, error_put_function
            )

    def de_register_dataset(self, dataset_id):
        with self.process_lock:
            del self._registered_datasets[dataset_id]
