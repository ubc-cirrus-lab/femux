import abc

class TransformerInterface(abc.ABC):
    @abc.abstractmethod
    def transform(preprocessed_df, timeseries_len):
        """
        Take requests per minute and transform to time series representation
        (e.g., concurrency, idle time...) at a given second.

        preprocessed_df: dataframe
        contains the  "HashFunction", "InvocationsPerMin", "ExecDurations", and
        "NumInvocations" columns.

            HashFunction: str
            the hash value associated with a function's invocation history

            HashApp: str
            hash value of the application that the function is associated with
                        
            InvocationsPerMin: list[int]
            the number of invocations per minute

            ExecDurations: list[int]
            average execution duration per day (1 for each day)

            NumInvocations: int
            total number of invocations

        num_minutes: int
        number of minutes that the input data spans

        returns: transformed_df 
        contains "HashFunction" and "TransformedValues" column.
        
            HashFunction: str
            the hash value associated with a function's invocation history

            TransformedValues: list[int or float]
            values of the new time series generated from transforming the 
            invocation data from the same row            
        
            NumInvocations: int
            total number of invocations
        """   
        pass