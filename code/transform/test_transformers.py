import unittest
import numpy as np
import pandas as pd
import unittest
from concurrency_transformer import ConcurrencyTransformer
from idletime_transformer import IdleTimeTransformer


class TestTransformers(unittest.TestCase):
    @classmethod
    def setUp(self):
        sample_list = np.array([704, 61])
        num_invocations = sum(sample_list)
        sample_duration = np.array([461])

        self.num_minutes = 765
        self.test_df = pd.DataFrame({"HashFunction": "abcd", "HashApp": "efgh", "InvocationsPerMin": [sample_list], "ExecDurations": [sample_duration], "NumInvocations": [num_invocations]})
        

    def test_conc_transform(self):
        test = ConcurrencyTransformer()

        result = test.gen_app_events(self.test_df, self.num_minutes)

        print(list(result.AppConcurrencyEvents.tolist()[0]))

                
    def test_app_conc(self):
        event_list_0 = [[0.5, 1], [0.6, 0]]
        event_list_1 = [[0.4, 1], [0.6, 0]]
        event_list_2 = [[0.5,2], [0.6, 1], [0.7,0]]


        expected_result = [(0.4, 1), (0.5, 2), (0.6, 1), (0.7, 0)]
        event_lists = [event_list_0, event_list_1, event_list_2]

        test = ConcurrencyTransformer()

        result = test.gen_app_concurrency_events(event_lists, 1, 4)

        print(result)       

    def test_app_conc_max_stays_at_one(self):
        event_list_0 = [[0.5, 1]]
        event_list_1 = [[0.4, 1]]
        event_list_2 = [[0.5,2], [0.6, 1], [0.7,0]]


        expected_result = [(0.4, 1), (0.5, 2), (0.6, 1)]
        event_lists = [event_list_0, event_list_1, event_list_2]

        test = ConcurrencyTransformer()

        result = test.gen_app_concurrency_events(event_lists, 1, 3)

        print(result)
    

    def test_idletime_minute_granularity(self):
        event_list = [[100,1], [200,2], [300,3], [400,2], [500,1], [600,2], [700,3], [800,2], [900,1], [1000,0]]
        expected_result = [3,3]

        test = IdleTimeTransformer()
        result = test.gen_app_idletimes(event_list)
        
        np.testing.assert_equal(expected_result, (result))


    def test_idletime_concurrency_jumps(self):
        event_list = [[100,1], [200,3], [300,4], [400,2], [500,1], [600,2], [700,3]]
        expected_result = [3,5]

        test = IdleTimeTransformer()
        result = test.gen_app_idletimes(event_list)
        
        np.testing.assert_equal(expected_result, (result))

    
    def test_multi_min_concurrency(self):
        df = pd.read_pickle("../data/transformed_data/concurrency/medium_app_conc_00.pckl")
        df = df.head(1)
        new_timestep = 5

        before_conversion = df.TransformedValues.tolist()[0]
        
        test = ConcurrencyTransformer()

        df["TransformedValues"] = df.TransformedValues.apply(lambda x : test.multi_minute_concurrency(x, 
                                                                    20160, new_timestep))
        
        after_conversion = df.TransformedValues.tolist()[0]
        
        np.testing.assert_almost_equal(sum(before_conversion[:new_timestep]) / new_timestep, after_conversion[0])



if __name__ == '__main__':
    unittest.main()