"""
Classes that represent hypnograms in either a 'sparse' or 'dense' format.
See SparseHypnogram and DenseHypnogram docstrings below.
"""

import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Union
from psg_utils import Defaults
from psg_utils.time_utils import TimeUnit, convert_time, standardize_time_input

logger = logging.getLogger(__name__)


class DenseHypnogram(pd.DataFrame):
    """
    Small wrapper around pd.DataFrame to represent hypnogram in dense representation
    Handles casting from/to specified TimeUnits and then initializes a typical dataframe with two columns:

        period_init_time | sleep_stage
    0   0                  0
    1   30                 0
    2   60                 1
        ...

    Where one sleep stage stored for each period of 'period_length' TimeUnits
    The period_length=30 in example above.
    See SparseHypnogram and SparseHypnogram.to_dense docstrings for details).
    Gaps are not allowed.
    """
    def __init__(self,
                 dense_init_times: Union[List[Union[int, float]], np.ndarray],
                 dense_stages: Union[List[int], np.ndarray],
                 time_unit: TimeUnit,
                 internal_time_unit: TimeUnit = TimeUnit.SECOND):
        """
        TODO
        """
        # Convert times to internal units
        dense_init_times = list(map(lambda t: convert_time(t, time_unit, internal_time_unit), dense_init_times))
        diffs = np.diff(dense_init_times)
        if np.any(diffs != diffs[0]):
            raise ValueError("Cannot initialize a DenseHypnogram with 'dense_init_times' "
                             "that contain gaps or unequal durations.")
        if np.any(np.asarray(dense_init_times) < 0):
            raise ValueError("Cannot initialize a DenseHypnogram with "
                             "'dense_init_times' that contain negative values.")

        # Init the DataFrame base object
        super(DenseHypnogram, self).__init__(data={
            "period_init_time": dense_init_times,
            "sleep_stage": dense_stages
        })


class SparseHypnogram(object):
    """
    Data structure for hypnogram internally represented sparsely by 3 lists:
    - init times (list of integers of initial period time points)
    - durations times (list of integers of seconds of period duration)
    - sleep stage (list of sleep stages, integer, for each period)

    Implements methods to query a sleep stage at a given time point from the
    sparse representation.
    """
    def __init__(self,
                 init_times: Union[List[Union[int, float]], np.ndarray],
                 durations: Union[List[Union[int, float]], np.ndarray],
                 sleep_stages: Union[List[int], np.ndarray],
                 period_length: [int, float],
                 time_unit: Union[TimeUnit, str] = TimeUnit.SECOND,
                 internal_time_unit: [TimeUnit, str] = TimeUnit.MILLISECOND):

        if not (len(init_times) == len(durations) == len(sleep_stages)):
            raise ValueError("Lists 'inits' and 'sleep_stages' must be of equal length.")

        # Convert times to internal (integer) representation
        self.time_unit = standardize_time_input(internal_time_unit)
        try:
            self.period_length = convert_time(period_length, time_unit, self.time_unit, cast_to_int=True)
        except ValueError as e:
            raise ValueError(f"Parameter 'period_length' should be a whole number/integer. "
                             f"Consider setting different org and/or internal time units "
                             f"(e.g., if you want to use a period_length of 2.5 milliseconds, "
                             f"set period_length=2.5, time_unit=TimeUnit.MILLISECONDS and "
                             "internal_time_unit=TimeUnit.MICROSECONDS.") from e
        try:
            init_times = list(map(lambda t: convert_time(t, time_unit, self.time_unit, cast_to_int=True), init_times))
            durations = list(map(lambda t: convert_time(t, time_unit, self.time_unit, cast_to_int=True), durations))
        except ValueError as e:
            raise ValueError("One of parameters 'inits' or 'durations' contains values which cannot be safely "
                             "represented as integers after time conversion. Consider specifying the 'time_unit' "
                             "and/or 'internal_time_unit' arguments to SparseHypnogram. E.g., if your hypnogram file "
                             "contains values in seconds such as 2.5, set time_unit=TimeUnit.SECOND and "
                             "internal_time_unit=TimeUnit.MILLISECOND to represent the 2.5 seconds as integer value "
                             "2500 internally.") from e

        if init_times[0] != 0:
            # Insert leading UNKNOWN class if hypnogram does not start at
            # second 0
            init_times.insert(0, 0)
            durations.insert(0, init_times[1])
            sleep_stages = list(sleep_stages)
            sleep_stages.insert(0, Defaults.UNKNOWN[1])

        self.inits = np.array(init_times, dtype=np.int64)
        self.durations = np.array(durations, dtype=np.int64)
        self.stages = np.array(sleep_stages, dtype=np.uint8)

        # Check sorted init times
        if np.any(np.diff(self.inits) < 0):
            raise ValueError("Array of init times must be sorted.")
        # Check init times and durations match
        from psg_utils.hypnogram.utils import hyp_has_gaps
        if hyp_has_gaps(self.inits, self.durations):
            raise ValueError("Found one or more gaps in hypnogram which is not allowed with class SparseHypnogram. "
                             "Consider manually filling hypnogram gaps with e.g., the 'UNKNOWN' stage. "
                             "Gaps may be filled using the 'psg_utils.hypnogram.utils.fill_hyp_gaps' function.")
        # Check no zero durations
        if np.any(self.durations == 0):
            raise ValueError("One or more durations in hypnogram are of length 0. "
                             "All segments of stages must have a positive duration.")

        # Make compact representation where contiguous identical stages are merged into one longer
        # OBS: Must call this after checking for gaps with hyp_has_gaps.
        self.make_compact()

    def __str__(self):
        return "SparseHypnogram(start={}, end={}, " \
               "length={}, stages={}, period_length={}, time_unit={})".format(
            self.inits[0], self.end_time, self.total_duration, self.classes, self.period_length, self.time_unit
        )

    def __repr__(self):
        return str(self)

    @property
    def n_classes(self) -> int:
        """
        Returns the current number of unique classes. Note this value could
        change after i.e. stripping functions have been applied the hypnogram
        """
        return len(self.classes)

    @property
    def classes(self) -> np.ndarray:
        """ Returns the unique classes/stages of the hypnogram """
        return np.unique(self.stages)

    @property
    def n_periods(self) -> int:
        """
        Returns the number of periods of length self.period_length in the hypnogram.
        Note that this include any partial final period of less tham self.period_length length.
        """
        return int(np.ceil(self.total_duration / self.period_length))

    @property
    def period_length_sec(self) -> float:
        """
        Returns the period length in seconds as a float
        """
        return float(convert_time(self.period_length, self.time_unit, TimeUnit.SECOND))

    @property
    def end_time(self) -> int:
        """ Hypnogram end time in unit self.time_unit """
        return int(self.inits[-1] + self.durations[-1])  # cast np.int -> int

    @property
    def end_time_sec(self) -> float:
        """
        Hypnogram end time in seconds as a float.
        """
        return convert_time(self.end_time, self.time_unit, TimeUnit.SECOND)

    @property
    def last_period_start(self) -> int:
        """ Returns the time point at which the last period begins """
        reminder = self.end_time % self.period_length
        if reminder > 0:
            return self.end_time - reminder
        else:
            return self.end_time - self.period_length

    @property
    def last_period_start_sec(self) -> float:
        """
        Returns the time point at which the last period begins in seconds as a float.
        """
        return convert_time(self.last_period_start, self.time_unit, TimeUnit.SECOND)

    @property
    def total_duration(self) -> int:
        """
        Returns the total length of the hypnogram.
        Identical to self.end_time when inits[0] == 0 (currently, always the
        case)
        """
        return int(np.sum(self.durations))

    @property
    def total_duration_sec(self) -> float:
        """
        Returns the total duration in TimeUnit.SECOND as a float.
        """
        return convert_time(self.total_duration, self.time_unit, TimeUnit.SECOND)

    @property
    def is_compact(self) -> bool:
        """
        Returns whether the current hypnogram is 'compact', i.e., there are no duplicate contiguous identical stages

        Returns:
            bool, Whether the SparseHypnogram is current 'compact' or not.
        """
        return not np.any(np.diff(self.stages.astype(np.int32)) == 0)

    def make_compact(self):
        """
        Squeezes the current hypnogram in place so that consecutive similar stages are merged
        into single, longer stages

        OBS: Hypnogram must not have gaps for this to function as expected

        Returns:
            SparseHypnogram, self
        """
        if not self.is_compact:
            from psg_utils.hypnogram.utils import squeeze_events, hyp_has_gaps
            if hyp_has_gaps(self.inits, self.durations):
                raise ValueError("Cannot call SparseHypnogram.make_compact on hypnogram that contains gaps.")
            self.inits, self.durations, self.stages = map(np.asarray,
                                                          squeeze_events(self.inits, self.durations, self.stages))
        assert self.is_compact
        return self

    def _extend_end_time(self, new_end_time: int):
        length_diff = new_end_time - self.end_time
        self.inits = np.append(self.inits, [self.end_time], axis=0)
        self.durations = np.append(self.durations, [length_diff], axis=0)
        self.stages = np.append(self.stages, [Defaults.UNKNOWN[1]], axis=0)

    def _shorten_end_time(self, new_end_time: int):
        if new_end_time <= 0:
            raise ValueError(f"New end time {new_end_time} must be greater than 0 {self.time_unit}")
        init_ind = np.where(new_end_time > self.inits)[0][-1]
        self.inits = self.inits[:init_ind+1]
        self.stages = self.stages[:init_ind+1]
        self.durations = self.durations[:init_ind+1]
        self.durations[-1] -= self.end_time - new_end_time

    def set_new_end_time(self, new_end_time: [int, float], time_unit: TimeUnit):
        """
        Trim the hypnogram from the tail by setting a new (shorter) end-time.

        Args:
            new_end_time: (int, float) New time at which the hypnogram is trimmed/extended to end
            time_unit:    (TimeUnit)   TimeUnit object specifying the time unit of new_end_time
        """
        # Find index of the new end time
        new_end_time = convert_time(new_end_time, time_unit, self.time_unit, cast_to_int=True)
        if new_end_time > self.end_time:
            self._extend_end_time(new_end_time)
        else:
            self._shorten_end_time(new_end_time)

    def get_index_at_time(self, time: [int, float], time_unit: TimeUnit) -> int:
        """
        Returns the index into self.inits, self.durations, self.stages corresponding to a time point.
        The index returned is zero-indexed and increments on exactly the start of a new hypnogram section.

        E.g., query(4, seconds) : inits [0, 5, 10], durations [5, 5, 5] --> index 0
        E.g., query(5, seconds) : inits [0, 5, 10], durations [5, 5, 5] --> index 1

        Args:
            time:         (int, float) The time point at which to query the hypnogram for an index
            time_unit:    (TimeUnit) TimeUnit object specifying the time unit of 'time'

        Returns:
            (int) Index into self.inits, self.durations, self.stages corresponding to time point 'time'.
        """
        time = convert_time(time, time_unit, self.time_unit, cast_to_int=True)
        # Check if time is within bounds of study duration
        if time < 0:
            raise IndexError("Query time must be >= 0 (got {})".format(time))
        if time < self.inits[0]:
            raise IndexError("Query time out of bounds (got time {}, but "
                             "first init time point is {} ({}))".format(time,
                                                                        self.inits[0],
                                                                        self.time_unit))
        if time >= self.end_time:
            raise IndexError("Query time out of bounds (got time {}, but"
                             " study ends at time {}, "
                             "last period starts at time {}) ({})".format(time,
                                                                          self.end_time,
                                                                          self.last_period_start,
                                                                          self.time_unit))
        # Find index of time point
        ind = np.searchsorted(self.inits, time)
        if ind == len(self.inits) or self.inits[ind] != time:
            ind -= 1
        assert ind >= 0
        return int(ind)

    def get_period_at_time(self, time: [int, float], time_unit: TimeUnit, on_overlapping: str = "RAISE") -> int:
        """
        Returns the sleep stage for a given period of length self.period_length in which time point 'time' falls within
        in the hypnogram. If the period spans multiple classes, the 'on_overlapping' parameter controls the following
        behaviour:

            Raw hypnogram: |00001 11100 00000| (0 and 1 are class integers)
            Periods:       |-----|-----|-----| (each block represents 1 period)

            FIRST:         |  0  |  1  |  0  | (0 and 1 are class integers)
            LAST:          |  1  |  0  |  0  |
            MAJORITY:      |  0  |  1  |  0  |

        I.e., with on_overlapping='FIRST', returns the first class integer observed within a given period.
              with on_overlapping='LAST', returns the last class integer observed within a given period
              with on_overlapping='MAJORITY', returns the class integer which spans the majority of time of a period
              with on_overlapping='RAISE', raises an error if there is any overlap for a given period_length (default).

        Args:
            time:           (int, float) The time point at which to query the sleep stage
            time_unit:      (TimeUnit)   TimeUnit object specifying the time unit of 'time'
            on_overlapping: (string)     One of 'FIRST', 'LAST', 'MAJORITY'. Controls the behaviour when a descrete
                                         period of length self.period_length overlaps 2 or more different classes in the
                                         original hypnogram.

        Returns:
            (int) The sleep stage integer class representation for the period of length self.period_length
                  in which time point 'time' falls.
        """
        time = convert_time(time, time_unit, self.time_unit, cast_to_int=True)

        # Find the time point and index signifying the beginning and end of the queried period
        period_start_time = time - time % self.period_length
        period_end_time = period_start_time + self.period_length
        start_ind = self.get_index_at_time(period_start_time, self.time_unit)
        try:
            end_ind = self.get_index_at_time(period_end_time, self.time_unit)
        except IndexError:
            # Exactly/above end
            end_ind = len(self.inits) - 1
        else:
            if period_end_time == self.inits[end_ind]:
                end_ind -= 1

        stages = list(self.stages[start_ind:end_ind+1])
        durations = list(self.durations[start_ind:end_ind+1])
        durations[0] = self.inits[start_ind] + self.durations[start_ind] - period_start_time
        durations[-1] = np.min([durations[-1] - np.sum(durations) + self.period_length, durations[-1]])

        # Find the stage depending on the requested overlap handling method (see __doc__).
        on_overlapping = on_overlapping.upper()
        if on_overlapping == "FIRST":
            return int(stages[0])
        elif on_overlapping == "LAST":
            return int(stages[-1])
        elif on_overlapping == "MAJORITY":
            return int(stages[np.argmax(durations)])
        elif on_overlapping == "RAISE":
            unique_stages = list(set(stages))  # Allow duplicates
            if len(unique_stages) == 1:
                return int(unique_stages[0])
            else:
                raise ValueError(f"Found {len(stages)} overlapping hypnogram annotations within requested period at "
                                 f"time point {time} ({self.time_unit}, period_length={self.period_length}) in period "
                                 f"[{period_start_time}, ..., {period_end_time}[ containing "
                                 f"{len(unique_stages)} different stages (labels: {unique_stages}). This is not "
                                 f"allowed as the 'on_overlapping' argument is set to 'RAISE'. If you want to allow "
                                 f"generating fixed-length contiguous stages from this hypnogram and this "
                                 f"period_length, set 'on_overlapping' to 'FIRST', 'LAST' or 'MAJORITY' "
                                 f"(see SparseHypnogram.get_period_at_time.__doc__) for details.")
        else:
            raise ValueError(f"Argument 'on_overlapping' must be one of 'FIRST', 'LAST', 'MAJORITY', 'RAISE' "
                             f"got '{on_overlapping}'")

    def get_stage_at_time(self, time: [int, float], time_unit: TimeUnit) -> int:
        """
        Returns the sleep stage at a given time point in the hypnogram

        Args:
            time:         (int, float) The time point at which to query the sleep stage
            time_unit:    (TimeUnit) TimeUnit object specifying the time unit of 'time'

        Returns:
            (int) The sleep stage integer class at the given time
        """
        return int(self.stages[self.get_index_at_time(time, time_unit)])

    def get_class_durations(self) -> defaultdict:
        """
        Computes the class counts for the hypnogram.

        Returns:
            A dictionary mapping class labels to counts.
        """
        counts = defaultdict(int)
        for stage, dur in zip(self.stages, self.durations):
            counts[stage] += dur
        return counts

    def to_dense(self, on_overlapping='RAISE', dense_time_unit: TimeUnit = TimeUnit.SECOND) -> DenseHypnogram:
        """
        Returns a DenseHypnogram representation of the stored data.

        Args:
            on_overlapping: (string)     One of 'FIRST', 'LAST', 'MAJORITY'. Controls the behaviour when a descrete
                                         period of length self.period_length overlaps 2 or more different classes in the
                                         original hypnogram. See self.get_period_at_time for details.
            dense_time_unit:  (TimeUnit) TimeUnit object specifying the time unit to use within the DenseHypnogram.

        Returns:
            A DenseHypnogram object
        """
        # Get all period start points
        period_start_points = np.arange(0, self.last_period_start+self.period_length, self.period_length)
        stages = [self.get_period_at_time(time, self.time_unit, on_overlapping) for time in period_start_points]
        return DenseHypnogram(dense_init_times=period_start_points,
                              dense_stages=stages,
                              time_unit=self.time_unit,
                              internal_time_unit=dense_time_unit)
