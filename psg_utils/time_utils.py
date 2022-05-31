import enum
import math


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


def convert_time(time: [float, int], from_unit: TimeUnit, to_unit: TimeUnit, cast_to_int=False):
    factor = to_unit.value / from_unit.value
    converted = time * factor
    if cast_to_int:
        if not math.isclose(converted, int(converted)):
            raise ValueError(f"Cannot safely cast time {time} converted "
                             f"to {converted} ({from_unit} -> {to_unit}) to integer value.")
        else:
            converted = int(converted)
    return converted
