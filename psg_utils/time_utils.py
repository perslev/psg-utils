import enum
import math


class TimeUnit(enum.Enum):
    SECOND = 1
    MILLISECOND = 10**3
    MICROSECOND = 10**6
    NANOSECOND = 10**9


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
