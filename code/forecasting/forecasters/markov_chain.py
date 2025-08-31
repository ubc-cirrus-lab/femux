import sys
import random
import time
import os
from numba import jit
import numpy as np

# sys.path.append("../../../../plotters")
# from superimposed_ftrace_plot import SuperImposedFtracePlot
# from mae_plotter import MaePlot
# from single_forecast_plot import SingleForecastPlot

# def distance_calculator_rmse(state_a, state_b):
#     rmse = 0
#     for i in range(len(state_a)):
#         rmse += (state_a[i] - state_b[i])**2
#     return (rmse / len(state_a)) ** (0.5)


@jit
def trace_scaling(trace):
        scaled_trace = np.zeros(len(trace))
        for i in range(len(trace)):
            if np.isclose(trace[i], 0):
                scaled_trace[i] = 0
            elif trace[i] < 1:
                scaled_trace[i] = round(trace[i], 7)
            else:
                scaled_trace[i] = round(trace[i], 2)
        return scaled_trace


class MarkovChainRadical:
    def __init__(self, trainingSet) -> None:
        self.max_pod_change_pos = 1
        self.max_pod_change_neg = 1
        self.max_frequency = 1
        self.local_freq = 1
        self.max_pod_count = 1
        self.same_pod_change_frequency = 1
        self.curr_sign = 0
        self.frequencies = [1]
        self.states = []
        self.all_vector_states = []
        # detrend for mc
        # quantize values in the trace to reduce number of states
        # detrendTrainingSet = [self.trace_scaling(elem) for elem in detrendTrainingSet]
        scaled_detrendTrainingSet = trace_scaling(trainingSet)
        self.trainMarkovChain(scaled_detrendTrainingSet)
        # important to reset this after training and before forecasting
        self.default_change_sign()
        self.min_elements_for_one_state = 2
        self.isDirty = False
        self.pvs = [0, 0, 0, 0]

        self.state_transition_adjacency_list = [{} for _ in range(len(self.all_vector_states))]
        self.state_transition_sum_list = [0 for _ in range(len(self.all_vector_states))]
        
        self.update_stal_stsm_with_training_states(scaled_detrendTrainingSet)
        self.first_time_after_training = True

    def default_change_sign(self):
        self.curr_sign = 0

    def update_state_transition_adjacency_list(self, prev_state_index, new_state_index):
        if new_state_index in self.state_transition_adjacency_list[prev_state_index]:
            self.state_transition_adjacency_list[prev_state_index][new_state_index] += 1
        else:
            self.state_transition_adjacency_list[prev_state_index][new_state_index] = 1
        self.state_transition_sum_list[prev_state_index] += 1

    def update_stal_stsm_with_training_states(self, trainingSet):
        training_trace = trainingSet
        prev_vector_state = self.vectorize_states(training_trace[0:self.min_elements_for_one_state])
        for i in range(self.min_elements_for_one_state-1, len(training_trace)-self.min_elements_for_one_state):
            new_vector_state = self.vectorize_states(training_trace[i:i+self.min_elements_for_one_state])
            self.update_state_transition_adjacency_list(
                self.all_vector_states.index(prev_vector_state),
                self.all_vector_states.index(new_vector_state)
            )
            self.pvs = prev_vector_state
            prev_vector_state = new_vector_state

    # If row sum is 0 then we get all rows which have non-zero elements
    # get the min sum across rows
    # divide each sum by the min sum
    # select a random number and select the index where the count is >= random no.
    def rowSumIsZero(self, stal, stsl):
        list_of_filled_rows = []
        sum_of_filled_rows = []
        dict_of_index_and_sum = {}
        for i in range(len(stal)):
            if stsl[i] != 0:
                list_of_filled_rows.append(i)
                sum_of_filled_rows.append(stsl[i])
        if len(sum_of_filled_rows) != 0:
            min_sum = min(sum_of_filled_rows)
            for j in range(len(sum_of_filled_rows)):
                sum_of_filled_rows[j] = round(sum_of_filled_rows[j] / min_sum)
                dict_of_index_and_sum[list_of_filled_rows[j]] = sum_of_filled_rows[j]
            dict_of_index_and_sum = {
                k: v
                for k, v in sorted(
                    dict_of_index_and_sum.items(), key=lambda item: item[1]
                )
            }
            rand = random.randint(1, max(sum_of_filled_rows))
            for k in range(len(dict_of_index_and_sum.items())):
                if dict_of_index_and_sum[list_of_filled_rows[k]] >= rand:
                    return list_of_filled_rows[k]
            print("this shouldnt happen1")
        else:
            print("this shouldnt happen2")

    # If row sum is not 0 then we get the sum of the row
    # find a random no. between 1 and the sum
    # select the index where the accumulated sum exceeds the random no.
    def rowSumIsNotZero(self, non_zero_row, non_zero_row_sum):
        agg_sum = non_zero_row_sum
        rand = random.randint(1, agg_sum)
        sum_so_far = 0
        for index in non_zero_row.keys():
            sum_so_far += non_zero_row[index]
            if sum_so_far >= rand:
                return index
        print("this shouldnt happen3")

    def check_difference_in_pods(self, trace):
        for state in self.states:
            if trace[-1] - trace[-2] == state:
                return state
        self.states.append(trace[-1] - trace[-2])
        self.states.sort()
        return trace[-1] - trace[-2]

    def check_scale_sign(self, trace):
        if trace[-1] > trace[-2]:
            self.curr_sign = 1
        elif trace[-1] < trace[-2]:
            self.curr_sign = -1
        return self.curr_sign

    def check_current_change_frequency(self, trace):
        if trace[-1] == trace[-2]:
            self.same_pod_change_frequency += 1
        else:
            self.same_pod_change_frequency = 1
        if self.same_pod_change_frequency not in self.frequencies:
            self.frequencies.append(self.same_pod_change_frequency)
        return self.frequencies[self.same_pod_change_frequency-1]

    def check_real_pod_count(self, trace):
        return trace[-1]

    def vectorize_states(self, trace):
        pod_count_state = self.check_real_pod_count(trace)  # P
        frequency_state = self.check_current_change_frequency(trace)  # I
        scale_state = self.check_difference_in_pods(trace)  # D
        scale_sign = self.check_scale_sign(trace)
        return [scale_state, frequency_state, pod_count_state, scale_sign]

    def forecast(self, forecast_window, prev_vector_state):
        i = 0
        index_of_current_state = self.all_vector_states.index(prev_vector_state)
        ftrace = []
        while i < forecast_window:
            if (
                self.state_transition_sum_list[index_of_current_state]
                == 0
            ):
                vector_element = self.all_vector_states[
                    self.rowSumIsZero(self.state_transition_adjacency_list, self.state_transition_sum_list)
                ]
            else:
                vector_element = self.all_vector_states[
                    self.rowSumIsNotZero( 
                        self.state_transition_adjacency_list[index_of_current_state],
                        self.state_transition_sum_list[index_of_current_state]
                    )
                ]

            # using pod count
            new_element_1 = vector_element[2]

            # using pod change
            new_element_2 = prev_vector_state[2] + vector_element[0]
            new_element = new_element_1

            prev_vector_state = vector_element

            ftrace.append(new_element)
            index_of_current_state = self.all_vector_states.index(vector_element)
            i += 1
        return ftrace

    def trainMarkovChain(self, trace):
        for i in range(len(trace)-1):
            pod_frequency = self.discrete_frequency(trace, i)
            # pod_change = self.discrete_pod_change(trace, i)
            # pod_count = self.discrete_pod_count(trace, i)
            # inlining the above two functions for efficiency
            pod_change = trace[i+1] - trace[i]
            pod_count = trace[i+1]
            sign_change = self.discrete_scale_sign(trace, i)
            new_vector_state = [pod_change, pod_frequency, pod_count, sign_change]
            if new_vector_state not in self.all_vector_states:
                self.all_vector_states.append(new_vector_state)


    def discrete_frequency(self, trace, i):
        if trace[i] == trace[i + 1]:
            self.local_freq += 1
        else:
            self.local_freq = 1
        if self.local_freq not in self.frequencies:
            self.frequencies.append(self.local_freq)
        return self.local_freq

    # def discrete_pod_change(self, trace, i):
    #     return trace[i+1] - trace[i] 
    
    # def discrete_pod_count(self, trace, i):
    #     return trace[i+1]
    
    def discrete_scale_sign(self, trace, i):
        if trace[i+1] > trace[i]:
            self.curr_sign = 1
        elif trace[i+1] < trace[i]:
            self.curr_sign = -1
        return self.curr_sign

    def check_max_frequency(self, trace):
        local_freq = 1
        for i in range(len(trace) - 1):
            if trace[i] == trace[i + 1]:
                local_freq += 1
            else:
                local_freq = 1
            self.max_frequency = max(self.max_frequency, local_freq)
        return self.max_frequency

    def check_max_pod_change(self, trace):
        local_pod_change_pos = 0
        local_pod_change_neg = 0
        for i in range(len(trace) - 1):
            if trace[i + 1] - trace[i] > 0:
                local_pod_change_pos = trace[i + 1] - trace[i]
            elif trace[i + 1] - trace[i] < 0:
                local_pod_change_neg = trace[i] - trace[i + 1]
            self.max_pod_change_pos = max(
                self.max_pod_change_pos, max(1, local_pod_change_pos)
            )
            self.max_pod_change_neg = max(
                self.max_pod_change_neg, max(1, local_pod_change_neg)
            )

    def check_max_pod_count(self, trace):
        for i in range(len(trace)):
            self.max_pod_count = max(self.max_pod_count, trace[i])

    def add_row_and_sum_to_stal_stsl(self):
        self.state_transition_adjacency_list.append({})
        self.state_transition_sum_list.append(0)

    def closest_state_calculator(self, new_vector_state):
        closest_vector_state = []
        least_distance = sys.maxsize
        com_thresh_rmse = least_distance
        for each_vector_state in self.all_vector_states:
            # distance = distance_calculator_rmse(each_vector_state, new_vector_state)
            rmse = 0
            rmse_calc_halted = False
            for i in range(len(each_vector_state)):
                rmse += (each_vector_state[i] - new_vector_state[i])**2
                if rmse > com_thresh_rmse:
                    rmse_calc_halted = True
                    break
            if rmse_calc_halted:
                continue
            distance = (rmse / len(each_vector_state)) ** (0.5)
            #
            if distance < least_distance:
                closest_vector_state = each_vector_state
                least_distance = distance
                com_thresh_rmse = rmse
        return closest_vector_state

    def distance_calculator_mae(self, state_a, state_b):
        mae = 0
        for i in range(len(state_a)):
            mae += abs(state_a[i] - state_b[i])
        return mae / len(state_a)

    # def trace_scaling(self, elem):
    #     if np.isclose(elem, 0):
    #         return 0
    #     elif elem < 1:
    #         return round(elem, 7)
    #     else:
    #         return round(elem, 2)

    def Forecast(self, trace, forecast_window): # the trace should have at least 3 elements
        # trace = [self.trace_scaling(elem) for elem in trace]
        scaled_trace = trace_scaling(trace)
        scaled_trace = scaled_trace[-3:]
        prev_vector_state = self.pvs
        new_vector_state = self.vectorize_states(scaled_trace)
        if new_vector_state not in self.all_vector_states:
            unknown_vector_state = new_vector_state
            new_vector_state = self.closest_state_calculator(new_vector_state)
            self.isDirty = True
        if not self.isDirty:
            if not self.first_time_after_training:
                self.update_state_transition_adjacency_list(
                    self.all_vector_states.index(prev_vector_state),
                    self.all_vector_states.index(new_vector_state),
                )
            self.pvs = new_vector_state
        ftrace = self.forecast(forecast_window, new_vector_state)
        if self.isDirty:
            self.all_vector_states.append(unknown_vector_state)
            self.add_row_and_sum_to_stal_stsl()
            self.update_state_transition_adjacency_list(
                self.all_vector_states.index(prev_vector_state),
                self.all_vector_states.index(unknown_vector_state),
            )
            self.pvs = unknown_vector_state
        self.isDirty = False
        self.first_time_after_training = False
        return ftrace
