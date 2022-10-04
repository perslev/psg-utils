"""
Functions for loading various types of hypnogram/sleep stage/labels file
formats from disk. Returns raw data either as a numpy array or
StartDurationStageFormat tuples.

Functions in utime.io.extractors.hyp_extractors will convert these data types
into utime.hypnogram objects which are used for all downstream operations.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from psg_utils import Defaults
from psg_utils.hypnogram.formats import StartDurationStageFormat
from psg_utils.hypnogram.utils import sparse_hypnogram_from_ids_format, ndarray_to_ids_format, squeeze_events
from psg_utils.time_utils import TimeUnit


def extract_from_edf(file_path, **kwargs):
    """
    Loader for hypnogram stored in EDF files in the EDF Annotations channel.
    Uses BaseEDFReader from .dhedreader to extract the data as
    Start-Duration-Stage lists. Returns data of type StartDurationStageFormat.

    See utime.hypnogram.formats.StartDurationStageFormat

    Returns:
        A StartDurationStageFormat object
    """
    from .dhedreader import BaseEDFReader
    with open(file_path, "rb") as in_f:
        # Get raw header
        base_edf = BaseEDFReader(in_f)
        base_edf.read_header()
        ann = tuple(zip(*tuple(base_edf.records())[0][-1]))
    return StartDurationStageFormat(ann)


def extract_from_start_dur_stage(file_path, **kwargs):
    """
    Loader for CSV-like files that store hypnogram information in the
    Start-Duration-Stage format.
    See utime.hypnogram.formats.StartDurationStageFormat

    Returns:
        A StartDurationStageFormat object
    """
    df = pd.read_csv(file_path, header=None)
    return StartDurationStageFormat(zip(*df.to_numpy()))


def extract_from_xml(file_path, **kwargs):
    """
    Extracts hypnograms from NSRR XML formatted annotation files.

    Returns:
        A StartDurationStageFormat object
    """
    import xml.etree.ElementTree as ET
    events = ET.parse(file_path).findall('ScoredEvents')
    assert len(events) == 1
    stage_dict = {
        "Wake|0": "W",
        "Stage 1 sleep|1": "N1",
        "Stage 2 sleep|2": "N2",
        "Stage 3 sleep|3": "N3",
        "Stage 4 sleep|4": "N3",
        "REM sleep|5": "REM",
        "Movement|6": "UNKNOWN",
        "Unscored|9": "UNKNOWN"
    }
    starts, durs, stages = [], [], []
    for event in events[0]:
        if not event[0].text == "Stages|Stages":
            continue
        stage = stage_dict[event[1].text]
        start = int(float(event[2].text))
        dur = int(float(event[3].text))
        starts.append(start)
        durs.append(dur)
        stages.append(stage)
    return StartDurationStageFormat((starts, durs, stages))


def extract_from_np(file_path, sample_rate, period_length, time_unit: TimeUnit = TimeUnit.SECOND):
    """
    Loader for hypnograms stored in numpy arrays (npz, npy).

    Returns:
        A StartDurationStageFormat object
    """
    arr = np.load(file_path)
    if not isinstance(arr, np.ndarray):
        # npz
        keys = list(arr.keys())
        assert len(keys) == 1
        arr = arr[keys[0]]
    return ndarray_to_ids_format(
        array=arr,
        period_length=period_length,
        time_unit=time_unit,
        sample_rate=sample_rate
    )


def extract_from_stg_txt(file_path, sample_rate, period_length, time_unit: TimeUnit = TimeUnit.SECOND):
    df = pd.read_csv(file_path, delimiter="\t")
    epoch, stages = df['Epoch'].values, df['User-Defined Stage'].values
    map_ = np.vectorize(
        {0: Defaults.AWAKE[0],
         1: Defaults.NON_REM_STAGE_1[0],
         2: Defaults.NON_REM_STAGE_2[0],
         3: Defaults.NON_REM_STAGE_3[0],
         4: Defaults.NON_REM_STAGE_3[0],
         5: Defaults.REM[0],
         6: Defaults.UNKNOWN[0],
         7: Defaults.UNKNOWN[0]}.get
    )
    # Map integer stages to default string values
    stages = map_(stages)
    # Insert UNKNOWN stages if there are gaps in 'epoch' list of epoch inds
    stages_proccessed = []
    for epoch_ind, stage in zip(epoch, stages):
        # Note: epoch_ind is 1 indexed, i.e. the first epoch has epoch ind '1' in file.
        n_missing = epoch_ind - (len(stages_proccessed) + 1)
        stages_proccessed.extend([Defaults.UNKNOWN[0]] * n_missing)
        stages_proccessed.append(stage)
    return ndarray_to_ids_format(
        array=stages,
        period_length=period_length,
        time_unit=time_unit,
        sample_rate=sample_rate
    )


def relative_time_stages_to_ids(relative_inits, durations, stages):
    # Filter Nones/False/empty from stages
    relative_inits, durations, stages_dense = filter_none_events(relative_inits, durations, stages)
    merged_inits, merged_durs, merged_stages = [], [], []
    for init, duration, stage in zip(relative_inits, durations, stages_dense):
        if merged_stages and (stage == merged_stages[-1] and (merged_inits[-1] + merged_durs[-1]) == init):
            # Continued stage, update last entry
            merged_durs[-1] += duration
        else:
            # New event
            merged_inits.append(init)
            merged_durs.append(duration)
            merged_stages.append(stage)
    return StartDurationStageFormat((merged_inits, merged_durs, merged_stages))


def correct_for_pauses(event_time, pauses):
    new_event_time = None
    for start, end in pauses:
        if event_time > start and event_time > end:
            # After pause, correct
            new_event_time = (new_event_time or event_time) - (end-start)
        elif event_time < start:
            # Before pause, skip
            continue
        else:
            # In middle of pause
            raise NotImplementedError(f"Not checked yet. {event_time}, {new_event_time} {pauses}")
    return new_event_time or event_time


def filter_none_events(start_times, durations, stages_dense):
    # Filter Nones/False/empty from stages
    filter_func = lambda s: s[-1] is not None and s[-1] != "None"
    start_times, durations, stages_dense = zip(*list(filter(filter_func, zip(start_times, durations, stages_dense))))
    return start_times, durations, stages_dense


def absolute_time_stages_to_relative_ids(start_times, durations, stages_dense, start_time, pause_periods=None):
    # TODO
    raise NotImplementedError("TODO")
    start_times, durations, stages_dense = filter_none_events(start_times, durations, stages_dense)
    relative_times, stages = [], []
    for event_index, (event_time, stage) in enumerate(zip(start_times, stages_dense)):
        if pause_periods:
            event_time = correct_for_pauses(event_time, pause_periods)
        diff_sec = (event_time - start_time).total_seconds()
        relative_times.append(diff_sec)
        stages.append(stage)
    return relative_time_stages_to_ids(
        relative_inits=relative_times,
        stages=stages,
        durations=durations
    )


def init_strings_to_datetime(inits, date_fmt, replace=None):
    dts = []
    for i in range(len(inits)):
        try:
            date = datetime.strptime(inits[i], date_fmt)
        except TypeError as e:
            print(inits[i], i, str(e))
            raise e
        if replace:
            date = date.replace(**replace)
        if i != 0:
            # Check if we passed midnight
            while (date - dts[i-1]).total_seconds() < 0:
                # Event time went to (one of) next day(s)
                date = date + timedelta(days=1)
        dts.append(date)
    return np.array(dts)


def extract_from_wsc_allscore(file_path, period_length_sec, sample_rate, event_date_fmt='%H:%M:%S.%f'):
    df = pd.read_csv(file_path, delimiter="\t", header=None, encoding="latin1")
    times, events = df[0].values, df[1].values
    times = init_strings_to_datetime(times, event_date_fmt)
    start_time = times[np.where(events == "START RECORDING")][0]
    pause_inds = np.where(events == "PAUSED")[0]
    resume_inds = np.where(events == "RECORDING RESUMED")[0]
    if len(pause_inds) and pause_inds[-1] == (len(events) - 1):
        # Remove last event which is a pause (effectively indicating end-of-file and thus have
        # no 'RECORDING RESUMED' event to be paired with)
        pause_inds = pause_inds[:-1]
    assert len(pause_inds) == len(resume_inds), "Different number of PAUSED and RECORDING RESUMED events found in file."
    pause_periods = list(zip(times[pause_inds], times[resume_inds]))
    map_ = np.vectorize(
        {"STAGE - W": Defaults.AWAKE[0],
         "STAGE - N1": Defaults.NON_REM_STAGE_1[0],
         "STAGE - N2": Defaults.NON_REM_STAGE_2[0],
         "STAGE - N3": Defaults.NON_REM_STAGE_3[0],
         "STAGE - R": Defaults.REM[0],
         "STAGE - NO STAGE": Defaults.UNKNOWN[0]}.get
    )
    stages = map_(list(map(lambda s: str(s).strip(), events)))
    inits, durs, stages = absolute_time_stages_to_relative_ids(
        start_times=times,
        stages_dense=stages,
        start_time=start_time,
        pause_periods=pause_periods,
        durations=[period_length_sec] * len(stages)
    )
    # Update durations to span from event index i to event index i+1
    for i in range(len(inits)-1):
        durs[i] = inits[i+1] - inits[i]
    return inits, durs, stages


def extract_from_stages_csv(file_path, period_length_sec, sample_rate, event_date_fmt='%H:%M:%S'):
    df = pd.read_csv(file_path,
                     sep=r'^([^,]+),[ ]?(\d+[.]?\d+)[ ]?,',
                     engine='python',
                     names=['Start Time', 'Duration (seconds)', 'Event'],
                     skiprows=1
                     )
    events = list(map(lambda s: str(s).strip(), df['Event'].values))
    try:
        # Look for "Start Recording" event.
        index = events.index('Beginning of Recording')
        start_date = init_strings_to_datetime([df.iloc[index, 0]], event_date_fmt)[0]
    except ValueError:
        import mne
        # Load the start time from EDF file
        edf_path = file_path.replace(".csv", ".edf")
        if not os.path.exists(edf_path):
            raise OSError("The hyp loader 'extract_from_stages_csv' requires an EDF file at path '{}' when processing "
                          "hypnogram in file '{}' in order to infer correct event start times relative to the EDF file. "
                          "However, no EDF file exists at the path.".format(edf_path, file_path))
        start_date = mne.io.read_raw_edf(edf_path, preload=False).info['meas_date']
        if not start_date:
            raise ValueError("Recording has no start time in EDF file at path '{}'. Cannot infer relative event start "
                             "times in event file '{}'".format(edf_path, file_path))
        assert isinstance(start_date, datetime)
    map_ = np.vectorize(
        {"Wake": Defaults.AWAKE[0],
         "Stage1": Defaults.NON_REM_STAGE_1[0],
         "Stage2": Defaults.NON_REM_STAGE_2[0],
         "Stage3": Defaults.NON_REM_STAGE_3[0],
         "REM": Defaults.REM[0],
         "UnknownStage": Defaults.UNKNOWN[0]}.get
    )
    stages = map_(events)
    return absolute_time_stages_to_relative_ids(
        start_times=init_strings_to_datetime(df['Start Time'].values, event_date_fmt, replace={
            'tzinfo': start_date.tzinfo,
            'day': start_date.day,
            'month': start_date.month,
            'year': start_date.year
        }),
        stages_dense=stages,
        start_time=start_date,
        durations=df['Duration (seconds)'].values
    )


def extract_from_nchsdb(file_path, period_length_sec, sample_rate):
    anots = pd.read_csv(file_path, delimiter="\t", names=['onset', 'duration', 'description'], header=0)
    onsets = anots['onset'].values
    durations = anots['duration'].values
    descriptions = list(anots['description'].values)
    map_ = np.vectorize({
                            "Sleep stage W": Defaults.AWAKE[0],
                            "Sleep stage N1": Defaults.NON_REM_STAGE_1[0],
                            "Sleep stage 1": Defaults.NON_REM_STAGE_1[0],
                            "Sleep stage N2": Defaults.NON_REM_STAGE_2[0],
                            "Sleep stage 2": Defaults.NON_REM_STAGE_2[0],
                            "Sleep stage N3": Defaults.NON_REM_STAGE_3[0],
                            "Sleep stage 3": Defaults.NON_REM_STAGE_3[0],
                            "Sleep stage R": Defaults.REM[0],
                            "Sleep stage ?": Defaults.UNKNOWN[0]
                        }.get)
    stages = map_(list(map(lambda s: str(s).strip(), descriptions)))
    try:
        start_recording = onsets[descriptions.index('Start Recording')]
    except ValueError:
        # Not annotated, assume 0.0 offset
        start_recording = 0.0

    return relative_time_stages_to_ids(
        relative_inits=[np.round(init - start_recording) for init in onsets],
        stages=stages,
        durations=durations
    )


_EXTRACT_FUNCS = {
    "edf": extract_from_edf,
    "sds": extract_from_start_dur_stage,
    "ids": extract_from_start_dur_stage,
    "xml": extract_from_xml,
    "npz": extract_from_np,
    "npy": extract_from_np,
    "stg.txt": extract_from_stg_txt,
    "wsc_allscore": extract_from_wsc_allscore,
    "STAGES": extract_from_stages_csv,
    "nchsdb": extract_from_nchsdb
}


def extract_ids_from_hyp_file(file_path,
                              period_length=None,
                              time_unit: TimeUnit = TimeUnit.SECOND,
                              sample_rate=None,
                              extract_func=None,
                              replace_zero_durations=False):
    """
    Entry function for extracing start-duration-stage format data from variable input files

    Args:
        file_path:      (str) Path to hypnogram file
        period_length:  (int, None) Only used for loading ndarray data.
                                    If None, ndarray must be dense (not signal-dense)
        time_unit:      (TimeUnit) The time unit for 'period_length' and inits/durations in file at 'file_path'

        sample_rate:    (int, None) Only used for loading ndarray data.
                                    If None, ndarray must be dense (not signal-dense)
        extract_func:   (callable, str, None) Callable or string identifier for callable as registered in _EXTRACT_FUNCS.
                                              If None, the file extension is used as string identifier, e.g. 'file.ids'
                                              will be loaded by the callable in _EXTRACT_FUNCS['ids'].
        replace_zero_durations: (bool, int/float): If not False replaces duration of length exactly 0.0 with this value.

    Returns:
        A StartDurationStageFormat object
    """
    if extract_func is None:
        extract_func = os.path.split(file_path)[-1].split('.', 1)[-1].lower()
    if not callable(extract_func):
        extract_func = _EXTRACT_FUNCS[extract_func]
    inits, durs, stages = extract_func(file_path=file_path,
                                       period_length=period_length,
                                       time_unit=time_unit,
                                       sample_rate=sample_rate)
    if replace_zero_durations:
        durs = np.where(np.isclose(durs, 0), replace_zero_durations, durs)
    return squeeze_events(inits, durs, stages)


def extract_hyp_data(file_path,
                     period_length,
                     annotation_dict,
                     sample_rate,
                     replace_zero_durations=False,
                     time_unit: TimeUnit = TimeUnit.SECOND,
                     hyp_internal_time_unit: TimeUnit = TimeUnit.MILLISECOND):
    """
    Load a hypnogram from a file at 'file_path'

    Args:
        file_path:              (str) A string path to the hypnogram file to load
        period_length:          (int, float) The length of 1 period of data in time units 'time_unit'.
        annotation_dict         (dict) Dictionary mapping from labels in array to sleep
        sample_rate:            (int) Sample rate of corresponding PSG data in Hz.
        replace_zero_durations: (bool, int/float): If not False replaces duration of length exactly 0.0 with this value.
        time_unit:              (TimeUnit) The time unit for 'period_length' and inits/durations in file at 'file_path'
        hyp_internal_time_unit  (TimeUnit) The time unit to use internally in the returned SparseHypnogram. Can usually
                                           be left at its default value.

    Returns:
        A SparseHypnogram object, annotation dict
    """
    ids_tuple = extract_ids_from_hyp_file(
        file_path,
        period_length=period_length,
        time_unit=time_unit,
        sample_rate=sample_rate,
        replace_zero_durations=replace_zero_durations
    )
    return sparse_hypnogram_from_ids_format(
        ids_tuple=ids_tuple,
        period_length=period_length,
        ann_to_class=annotation_dict,
        time_unit=time_unit,
        hyp_internal_time_unit=hyp_internal_time_unit
    )
