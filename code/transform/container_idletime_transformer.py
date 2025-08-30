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

class ContainerIdleTimeTransformer(TransformerInterface):   

    def transform(self, event_df, num_seconds):
        if event_df.shape[0] == 0:
            return event_df
        
        event_df.reset_index(drop=True, inplace=True)

        event_df["TransformedValues"] = event_df.apply(lambda x : self.gen_container_idletimes(x.TransformedValues, x.HashApp), axis=1)

        print("finished idletime transform for chunk", strftime("%H:%M:%S"))

        return event_df


    def gen_container_idletimes(self, avg_conc_per_min, hashapp):
        """Convert average concurrency into container idletimes, which are the 
        amount of minutes with an average concurrency of 0 (so no containers).

        avg_conc_per_min: list[float]
        list of events that store event time and concurrency respectively

        returns: np.array(int)
        number of minutes between each container
        """
        idletime_series = []
        first_invocation = True
        cur_idletime = 0

        for cur_minute in range(len(avg_conc_per_min)):
            if avg_conc_per_min[cur_minute] > 0:
                if not first_invocation:
                    idletime_series.append(cur_idletime)
                else:
                    first_invocation = False

                cur_idletime = 0
            else:
                cur_idletime += 1

        return np.array(idletime_series)
