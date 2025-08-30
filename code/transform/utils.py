import gc
import numpy as np
import pandas as pd

SECONDS_IN_MIN = 60
MILLISECONDS_PER_SECOND = 1000
MINUTES_PER_DAY = 1440
BATCHES_FOR_LARGE_INVOCATIONS = 600

def gen_events(invocation_counts, total_num_invocations, exec_durations):
    """Generate invocation and endtime events based on the invocations per minute.

        invocation_counts: list[tuple(int, float)]
        list containing tuples with number of invocations per minute and concurrency

        num_invocations: int
        total number of invocations 

        exec_durations: int
        number 

        returns: tuple(list[float], list[float])
        the list of invocation and endtimes respectively
        """

    # batch invocations if application has >100M invocations
    large_app = total_num_invocations > 100000000

    # some functions have an execution duration of 0ms, so we bump that up to 
    # 1us to separate start and endtimes
    exec_durations = [max(duration, 0.001) for duration in exec_durations]

    # batch events for large apps, so need to append items instead of using preset value 
    invocation_times = [] if large_app else np.empty(total_num_invocations, dtype='float64,f')
    invocation_endtimes = [] if large_app else np.empty(total_num_invocations, dtype='float64,f') 
    event_index = 0
    invocation_index_per_day = []
    cur_day = 0
    exec_duration = exec_durations[cur_day] / MILLISECONDS_PER_SECOND

    # for each minute we generate the events
    for cur_minute in range(len(invocation_counts)):
        cur_num_invocations = invocation_counts[cur_minute]

        # update values for new day
        if cur_day < (cur_minute // MINUTES_PER_DAY):
            cur_day += 1
            
            invocation_index_per_day.append(event_index)

            exec_duration = exec_durations[cur_day] / MILLISECONDS_PER_SECOND                   

        # no new invocations so we skip
        if cur_num_invocations == 0:
            continue

        # if there are less than 600 invocations per minute, then we capture each event,
        # otherwise we batch events together at a 100ms granularity (which is why we 
        # choose 600 batches within a minute). 100ms is on the low end of cold starts
        if large_app and cur_num_invocations > 600:
            iat = SECONDS_IN_MIN / BATCHES_FOR_LARGE_INVOCATIONS
            concurrency = cur_num_invocations / BATCHES_FOR_LARGE_INVOCATIONS
            num_events = BATCHES_FOR_LARGE_INVOCATIONS
        else:
            iat = SECONDS_IN_MIN / cur_num_invocations
            concurrency = 1
            num_events = cur_num_invocations

                    
        shift = iat / 2
        cur_second = cur_minute * SECONDS_IN_MIN
        cur_invocation_time = shift + cur_second

        # generate the events and store them in their respective lists
        for _ in range(num_events):
            if large_app:
                invocation_times.append((cur_invocation_time, concurrency))
                invocation_endtimes.append((cur_invocation_time + exec_duration, concurrency))
            else:
                invocation_times[event_index] = (cur_invocation_time, concurrency)
                invocation_endtimes[event_index] = (cur_invocation_time + exec_duration, concurrency)
            cur_invocation_time += iat
            event_index += 1


    # large applications will truncate these lists from batching invocations
    if large_app:
        invocation_times = np.array(invocation_times)
        invocation_endtimes = np.array(invocation_endtimes)

    # sort endtime inversions which can occur if previous day's duration is
    # longer than following day
    for i, day_inv_index in enumerate(invocation_index_per_day):
        # skip days with no invocations
        if day_inv_index == 0 or (i > 0 and day_inv_index == invocation_index_per_day[i-1]):
            continue

        # longer than the current day's
        if day_inv_index < event_index and invocation_endtimes[day_inv_index - 1][0] > invocation_endtimes[day_inv_index][0]:
            fix_inversions(invocation_endtimes, day_inv_index)

    return pd.Series([invocation_times, invocation_endtimes, event_index])


def fix_inversions(invocation_endtimes, cur_day_index):
    """Sort endtimes that occur at boundary between two days. Endtimes from the previous
    day can occur after endtimes from current day if execution duration from previous day
    is longer than current day.

    invocation_endtimes: np.array(float or int)
    list of endtimes

    cur_day_index: int
    index for the first endtime that occurs for the current day
    """
    start_index = cur_day_index - 1
    end_index = cur_day_index + 1

    # find first invocation that ends in the current day
    while invocation_endtimes[start_index][0] > invocation_endtimes[cur_day_index][0]:
        start_index -= 1

    # find first endtime from current day that occurs after the last endtime from previous day
    while end_index < len(invocation_endtimes) and invocation_endtimes[end_index][0] < invocation_endtimes[cur_day_index - 1][0]:
        end_index += 1


    insertion_sort(invocation_endtimes, start_index, end_index)


def insertion_sort(arr, start, end):
    """In-place insertion sort
    arr: np.arr(int or float)
    array to be sorted

    l: int
    start index of sub list to be sorted

    r: int
    end index of sub list to be sorted
    """
    for i in range(start + 1, end):
        cur = arr[i]
        j = i - 1

        while (j >= start) & (arr[j][0] > cur[0]):
            arr[j + 1] = arr[j]
            j -= 1

        arr[j + 1] = cur


def load_balance(val_list, chunks):
    val_list.sort(reverse=True)

    for val in val_list:
        min_index = 0
        min_sum = sum(chunks[0])

        for i, chunk in enumerate(chunks):
            if sum(chunk) < min_sum:
                min_index = i
                min_sum = sum(chunk)

        chunks[min_index].append(val)