import logging
import sys
import pandas as pd
import numpy as np
from transform.transformer_interface import TransformerInterface
from time import strftime
from transform.utils import gen_events

#logging.basicConfig(filename='transformers.log', encoding='utf-8', level=logging.DEBUG, filemode='w')
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

SECONDS_PER_MINUTE = 60
MINUTES_PER_DAY = 1440


class ConcurrencyTransformer(TransformerInterface):   
    def transform(self, event_df, num_seconds, timestep=60):
        """ Transform concurrency events to average concurrency per 
        timestep (e.g., 60 seconds)

        event_df: pd.DataFrame()
        columns:
            ConcurrencyEvents: list[(float, int)]
            an event marks the time and concurrency whenever concurrency changes

            HashApp: str
            name of application
        """
        if event_df.shape[0] == 0:
            return event_df
        
        event_df.reset_index(drop=True, inplace=True) 

        event_df[["TransformedValues", "ContainerInvocationsPerMin"]] = event_df.apply(lambda x : self.gen_concurrency_timeseries(
                                                        x.ConcurrencyEvents, num_seconds), axis=1)
        
        event_df.drop(columns=["ConcurrencyEvents"], inplace=True)

        event_df["TransformedValues"] = event_df.TransformedValues.apply(lambda x : self._gen_per_minute_concurrency(x,
                                            num_seconds, timestep))

        print("finished conc transform for chunk", strftime("%H:%M:%S"))
        return event_df


    def transform_concurrency_events(self, preproc_df, num_minutes, func_mode):
        
        if preproc_df.shape[0] == 0:
            print("skipping event transform for empty chunk", strftime("%H:%M:%S"))
            return preproc_df
        
        num_seconds = num_minutes * SECONDS_PER_MINUTE

        preproc_df.reset_index(drop=True, inplace=True)
        
        # generate invocation and endtime event lists
        preproc_df[["InvocationTimes", "InvocationEndTimes", 
                    "NumEvents"]] = preproc_df.apply(lambda x : gen_events(x.InvocationsPerMin,
                                                        x.NumInvocations, x.ExecDurations), axis=1)

        preproc_df.drop(columns=["InvocationsPerMin", "NumInvocations", "ExecDurations"], inplace=True, axis=1)

        # generate event list for each time concurrency changes
        preproc_df["ConcurrencyEvents"] = preproc_df.apply(lambda x : self.gen_concurrency_events(x.InvocationTimes,
                                                            x.InvocationEndTimes, x.NumEvents, num_seconds), axis=1)

        if not func_mode:
            # group rows based on HashApp
            preproc_df = preproc_df.groupby("HashApp").agg({"ConcurrencyEvents": list, "NumEvents": "sum"})

            preproc_df.reset_index(inplace=True)

            # combine concurrency events at function-level to get concurrency events at application level
            preproc_df[["ConcurrencyEvents", "NumEvents"]] = preproc_df.apply(lambda x : 
                                                                ConcurrencyTransformer.gen_app_concurrency_events(x.ConcurrencyEvents,
                                                                num_seconds, x.NumEvents), axis=1)

            preproc_df = preproc_df[["HashApp", "NumEvents", "ConcurrencyEvents"]]
        
        return preproc_df

    def gen_concurrency_events(self, invocation_times, invocation_endtimes, num_events, num_seconds):
        """Generate concurrency events based on the invocation and endtimes.

        event_lists: tuple(list[tuple(float, float)], list[tuple(float,float)])
        the lists of invocation and endtimes with their associated concurrencies

        num_events: int
        number of events contained in our event list

        num_seconds: int
        any event that occurs past this value will not be stored in the event list

        returns: list[tuple(float, int)]
        list of events that contain the event time and concurrency
        """
        # for traces with no invocations
        if len(invocation_times) == 0:
            return np.empty(0)

        endtime_index = 0
        invocation_index = 0
        concurrency = 0
        concurrency_event_list = np.empty(num_events*2, dtype='float64,f')
        next_invocation_time = invocation_times[0][0]
        next_endtime = invocation_endtimes[0][0]

        for event_index in range(num_events * 2):
            if invocation_index < num_events and next_invocation_time < next_endtime:
                concurrency += invocation_times[invocation_index][1]
                concurrency_event_list[event_index] = (next_invocation_time, concurrency)
                invocation_index += 1
                if invocation_index < num_events:
                    next_invocation_time = invocation_times[invocation_index][0]
            else:                
                concurrency -= invocation_endtimes[endtime_index][1]
                concurrency_event_list[event_index] = (next_endtime, concurrency)
                endtime_index += 1
                if endtime_index < num_events:
                    next_endtime = invocation_endtimes[endtime_index][0]
        
        # remove all events that happen outside the time horizon
        while concurrency_event_list[event_index][0] >= num_seconds:
            event_index -= 1

        return concurrency_event_list[:event_index + 1]

          
    def gen_concurrency_timeseries(self, event_list, num_seconds):
        """Convert concurrency event list into per-second average concurrency series.

        event_list: list[tuple(float, int)]
        list of events that store event time and concurrency respectively

        num_seconds:
        length of time series

        """
        num_minutes = num_seconds // SECONDS_PER_MINUTE
        concurrency_timeseries = np.zeros(num_seconds)
        container_invocations = np.zeros(num_minutes, dtype=int)

        if len(event_list) == 0:
            return pd.Series([concurrency_timeseries, container_invocations])

        stop_index = self._get_stop_index(event_list, num_seconds)

        # first event is always container invocations
        container_invocations[int(event_list[0][0] // SECONDS_PER_MINUTE)] = event_list[0][1]

        # get per-second average concurrency
        for event_index in range(stop_index):
            cur_event = event_list[event_index]
            next_event = event_list[event_index + 1]
            next_event_minute = int(next_event[0] // SECONDS_PER_MINUTE)


            if next_event[1] > cur_event[1] and next_event_minute < num_minutes:
                # batching means concurrency can jump by more than one
                container_invocations[next_event_minute] += int(next_event[1] - cur_event[1])

            self._add_to_concurrency_timeseries(concurrency_timeseries, cur_event[0], next_event[0], 
                                                cur_event[1], num_seconds)


        return pd.Series([concurrency_timeseries, container_invocations])


    def _add_to_concurrency_timeseries(self, concurrency_series, start_time, end_time, concurrency, num_seconds):
        """Add (end-start) * duration to the respective second quanta.

        concurrency_series: list[int]
        each element of the concurrency timeseries is the concurrency value within that second

        start_time: float
        time that invocation starts

        end_time: float
        time that invocation ends execution

        concurrency: int
        invocation concurrency during that time

        num_seconds: int
        length of time series
        """
        cur_time = start_time
        cur_sec = int(cur_time)
        next_sec = int(cur_time + 1)
        end_sec = min(int(end_time), num_seconds - 1)

        while cur_sec <= end_sec:

            if end_time > next_sec:
                concurrency_series[cur_sec] += concurrency * (next_sec - cur_time)

            else:
                concurrency_series[cur_sec] += concurrency * (end_time - cur_time)

            cur_sec += 1
            next_sec += 1
            cur_time = cur_sec


    def _get_stop_index(self, event_list, num_seconds):
        """Return index of first event that occurs after the last second of data 
        we are capturing.

        event_list: list[tuple(float, int)]
        list of events where each event stores the time and concurrency respectively

        num_seconds:
        length of our time series
        """
        last_event_index = len(event_list) - 1

        for stop_index in range(last_event_index, 0, -1):
            
            # return the first event that occurs after the end of our time series
            # or the last event if no such event exists.
            if event_list[stop_index][0] < num_seconds:
                return min(stop_index + 1, last_event_index)


    def _gen_per_minute_concurrency(self, sec_concurrency_series, num_seconds, timestep):
        """Convert a per-second average concurrency to larger timesteps.

        sec_concurrency_series: list[float]
        concurrency per second

        num_seconds: int
        length of time series

        timestep:
        size of new timestep in seconds
        """

        new_concurrency_series = np.zeros(num_seconds // timestep)
        cur_timestep = 0

        for cur_sec in range(num_seconds):
            if cur_sec // timestep > cur_timestep:
                cur_timestep += 1

            new_concurrency_series[cur_timestep] += sec_concurrency_series[cur_sec] 

        new_concurrency_series = new_concurrency_series / timestep

        return new_concurrency_series


    def multi_minute_concurrency(self, concurrency_series, num_minutes, new_timestep, mode):
        """Convert per-minute concurrency to a large timestemp (e.g., 5 or 10 minutes)

        concurrency_series: list[float]
        average concurrency per minute

        num_minutes: int
        number of minutes in total trace

        new_timestep: int
        size of the new timestep in minutes

        mode: str
        how the new timsteps are calculated
            "max": maximum value from the timesteps being merged
            "avg": average value from timesteps being merged
        """
        cur_timestep = 0
        cur_minute = 0
        timestep_val = 0
        new_concurrency_series = np.empty(num_minutes // new_timestep)

        for concurrency in concurrency_series: 
            if cur_minute // new_timestep > cur_timestep:
                if mode == "avg":
                    timestep_val /= new_timestep
                
                new_concurrency_series[cur_timestep] = timestep_val
                cur_timestep += 1
                timestep_val = 0
            
            timestep_val = self._multi_minute_metric(timestep_val, concurrency, mode)
            cur_minute += 1

        return new_concurrency_series


    def _multi_minute_metric(self, metric, new_val, mode):
        if mode == "avg":
            return new_val + metric
        elif mode == "max":
            return max(new_val, metric)

    @staticmethod
    def gen_app_concurrency_events(event_lists, num_seconds, num_events):
        """ Generate concurrency event list for the application based on the concurrency events 
        of each function. Application always takes the max concurrency out of all functions 
        at any point in time.

        event_lists: list[list[tuple(float, float)]]
        list of event lists where each event contains the event time and concurrency
        """        
        event_lists = [event_list for event_list in event_lists if len(event_list) > 0]

        num_event_lists = len(event_lists)
        
        # if there's just one event list then there is no combining to be done
        if num_event_lists == 1:
            return pd.Series([event_lists[0], num_events])
        elif num_event_lists == 0:
            return pd.Series([[], 0])

        event_indices = np.zeros(num_event_lists, dtype=int)
        event_times = np.array([event_list[0][0] for event_list in event_lists])

        end_indices = np.array([len(event_list) for event_list in event_lists])
        done_list = np.zeros(num_event_lists, dtype=int)
        func_concurrencies = np.zeros(num_event_lists, dtype=int)
        app_concurrency = 0
        app_event_index = 0
        app_events = np.empty(num_events*2, dtype='f,f')

        # update app-level concurrency until there is just one function left to update from
        while sum(done_list) < num_event_lists - 1:
            # get the next event
            func_index = np.argmin(event_times)
            event_index = event_indices[func_index]
            
            # get the event time and concurrency
            cur_event_time, cur_concurrency = event_lists[func_index][event_index]

            func_concurrencies[func_index] = cur_concurrency

            max_concurrency = max(func_concurrencies)

            # if max concurrency changed, store this event in the application event list
            if max_concurrency != app_concurrency:
                app_concurrency = max_concurrency
                app_events[app_event_index] = (cur_event_time, app_concurrency)
                app_event_index += 1
            

            """ update state """
            next_event_index = event_index + 1

            # if we reached last event of this function, mark it done and set its event time to the 
            # end of the time series (so the function won't be chosen again)
            if next_event_index == end_indices[func_index]:
                done_list[func_index] = 1
                event_times[func_index] = num_seconds
            else:
                event_indices[func_index] = next_event_index
                event_times[func_index] = event_lists[func_index][next_event_index][0]


        # now we have one remaining event list and add it to the end of the app list 
        last_func_index = np.argmin(done_list)
        other_conc = max(np.delete(func_concurrencies, last_func_index))
        last_func_event_index = event_indices[last_func_index]
        last_event_list = event_lists[last_func_index]
        remaining_events = last_event_list[last_func_event_index:]

        for remaining_event in remaining_events:
            max_concurrency = max(remaining_event[1], other_conc)
            if max_concurrency != app_concurrency:
                app_concurrency = max_concurrency
                app_events[app_event_index] = (remaining_event[0], max_concurrency)
                app_event_index += 1

        return pd.Series([app_events[:app_event_index], app_event_index])
