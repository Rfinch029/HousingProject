import numpy as np # provides the library for ndarray data structures (other libraries depend on this)

class LSTM:
    def __init__(self, input_dim, hidden_dim):
        # initializing the dimensions
        self.input_dim = input_dim # input_dimension is defined by the number of features
        self.hidden_dim = hidden_dim

