import enum
import math
from typing import Union, Tuple


class TimeUnit(enum.Enum):
    SECOND = 1
    MILLISECOND = 10**3
    MICROSECOND = 10**6
    NANOSECOND = 10**9

    @classmethod
    def from_string(cls, time_string: str):
        time_string = time_string.upper().rstrip('S')
        try:
            return getattr(cls, time_string)
        except AttributeError as e:
            raise AttributeError(f"Invalid time string '{time_string}' passed. "
                                 f"Valid options are SECOND(S), MILLISECOND(S), MICROSECOND(S), NANOSECOND(S) "
                                 f"in upper/lower case letters.") from e


def standardize_time_input(time_input: Union[TimeUnit, str]) -> TimeUnit:
    """
    Takes a time input 'time_input' of type TimeUnit or a string convertable with TimeUnit.from_string
    and returns a TimeUnit object.
    """
    if isinstance(time_input, TimeUnit):
        return time_input
    else:
        return TimeUnit.from_string(time_input)


def convert_time(time: [float, int], from_unit: Union[TimeUnit, str], to_unit: Union[TimeUnit, str], cast_to_int=False):
    to_unit = standardize_time_input(to_unit)
    from_unit = standardize_time_input(from_unit)
    factor = to_unit.value / from_unit.value
    converted = time * factor
    if cast_to_int:
        if not math.isclose(converted, round(converted)):
            raise ValueError(f"Cannot safely cast time {time} converted "
                             f"to {converted} ({from_unit} -> {to_unit}) to integer value.")
        else:
            converted = round(converted)
    return converted
