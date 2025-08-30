import logging
import sys
import numpy as np
from time import strftime
from utils import gen_events
from transformer_interface import TransformerInterface
from concurrency_transformer import ConcurrencyTransformer

#logging.basicConfig(filename='transformers.log', encoding='utf-8', level=logging.DEBUG, filemode='w')
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

MILLISECONDS_PER_SECOND = 1000
SECONDS_PER_MINUTE = 60
MINUTES_PER_DAY = 1440
RUNNING = 10000000000
IDLING = 100000000000

class InvocationIdleTimeTransformer(TransformerInterface):   

    def transform(self, event_df, num_minutes):
        if event_df.shape[0] == 0:
            return event_df
        
        
        event_df.reset_index(drop=True, inplace=True)

        event_df["TransformedValues"] = event_df.apply(lambda x : self.gen_app_idletimes(x.ConcurrencyEvents)
                                                        , axis=1)

        event_df.drop(columns=["ConcurrencyEvents"], inplace=True)

        print("finished idletime transform for chunk", strftime("%H:%M:%S"))

        return event_df


    def gen_app_idletimes(self, conc_events):
        """Convert concurrency event list into idletimes.

        conc_events: list[tuple(float, int)]
        list of events that store event time and concurrency respectively

        num_invocations:
        number of invocations captured in time series

        returns:
        """
        num_idletimes = len(conc_events) - 1
        idletime_series = np.empty(num_idletimes)
        instance_start_times = np.empty(0)
        instance_end_times = np.empty(0)
        idletime_series_index = 0
        num_instances = 0
        prev_concurrency = 0

        for cur_event in conc_events:
            event_time = cur_event[0]
            event_concurrency = cur_event[1]

            # we decrement/increment depending on the increase or decrease in the new events concurrency
            step = 1 if event_concurrency > prev_concurrency else -1

            for app_concurrency in range(prev_concurrency + step, event_concurrency + step, step):

                # add another instance
                if app_concurrency > num_instances:
                    num_instances += 1
                    instance_start_times = np.append(instance_start_times, event_time)
                    instance_end_times = np.append(instance_end_times, RUNNING)
                
                # start instance that has been idling the longest and set its state to running
                elif app_concurrency == prev_concurrency + 1:
                    next_instance_index = np.argmin(instance_end_times)
                    instance_start_times[next_instance_index] = event_time

                    # convert idletime to minute granularity
                    idletime = (event_time - instance_end_times[next_instance_index]) // SECONDS_PER_MINUTE
                    idletime_series[idletime_series_index] = 0 if idletime < 1 else idletime
                    instance_end_times[next_instance_index] = RUNNING
                    idletime_series_index += 1

                # scale down instance that has been running the longest and set its state to idle
                elif app_concurrency == prev_concurrency - 1:
                    scaledown_instance_index = np.argmin(instance_start_times)
                    instance_end_times[scaledown_instance_index] = event_time
                    instance_start_times[scaledown_instance_index] = IDLING
                else:
                    raise Exception("Concurrency has changed by more than 1")

                prev_concurrency = app_concurrency
        
        return idletime_series[:idletime_series_index]


    def _gen_idletimes(self, event_lists, num_invocations):
        """Generate idletime series based on invocation and endtimes.
        If the next invocation precedes the endtime of the current inovcation,
        we count that idletime as 0.

        event_lists: tuple(list[float], list[float])
        the list of invocation and endtimes respectively

        num_invocations:
        number of invocations contained in our event list
        """
        invocation_events = event_lists[0]
        endtime_events = event_lists[1]

        # leave out last idletime as there is no invocation that follows
        idletime_series = np.zeros(num_invocations - 1)

        # update every idletime that is nonzero
        for event_index in range(num_invocations - 1):
            next_invocation_time = invocation_events[event_index + 1]
            cur_endtime = endtime_events[event_index]

            if next_invocation_time > cur_endtime:
                idletime_series[event_index] = next_invocation_time - cur_endtime

        return idletime_series