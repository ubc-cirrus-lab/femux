import abc

class SimulatorInterface(abc.ABC):
    @abc.abstractmethod
    def run_sim(traces_df, num_workers, save_file_name):
        """
        traces: df[] 
        dataframe of traces with the following structure:
        col 1: function hash | col 2: time series | col 3: fast/slow tag

        col 1 name = HashFunction
        col 2 name = InvocationConcurrencyLists

        num_workers: int
        number of parallel processes to run

        save_file_name: str
        name of pickle file to write to (e.g., forecasted_data.pckl)

        output: save_file_name
        dump dataframe in a pckl file with new columns containing data from
        the simulations.
        """
        pass