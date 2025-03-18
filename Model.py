import numpy as np # provides the library for ndarray data structures (other libraries depend on this)

class LSTM:
    def __init__(self, input_dim, hidden_dim):
        # initializing the dimensions
        self.input_dim = input_dim # input_dimension is defined by the number of features
        self.hidden_dim = hidden_dim # Number of hidden neurons in model. Appropriate # --> balance of complexity

        # Xavier Initialization for weights
        # What is Xavier Initialization? The Xavier Initialization accounts for issues of exploding and shrinking
        # gradients by normalizing variance accordingly. "Basically, an additional safety measure"

        # NOTE: all weight and biases are defined as NumPy arrays

        # Weights and biases for forget gate
        self.W_f = np.random.randn(hidden_dim, input_dim + hidden_dim) * np.sqrt(1 / input_dim + hidden_dim)
        self.b_f = np.zeros((hidden_dim, 1))

        # Weights and biases for input gate
        self.W_i = np.random.randn(hidden_dim, input_dim + hidden_dim) * np.sqrt(1 / input_dim + hidden_dim)
        self.b_i = np.zeros((hidden_dim, 1))

        # Weights and biases for cell-state gate
        self.W_c = np.random.randn(hidden_dim, input_dim + hidden_dim) * np.sqrt(1 / input_dim + hidden_dim)
        self.b_c = np.zeros((hidden_dim, 1))

        # Weights and biases for output gate
        self.W_o = np.random.randn(hidden_dim, input_dim + hidden_dim) * np.sqrt(1 / input_dim + hidden_dim)
        self.b_o = np.zeros((hidden_dim, 1))

    # activation functions
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2

    # forward pass method.
    def forward(self, x):
        T = x.shape[0]  # T represents the length of the input sequence
        h = np.zeros((self.hidden_dim, 1))  # Initialize hidden state
        c = np.zeros((self.hidden_dim, 1))  # Initialize cell state

        self.cache = []  # stores intermediate values for back-propagation
        h_seq = []  # stores hidden states as they are updated

        for t in range(T):
            xt = x[t].reshape(-1, 1)  # reshape input at time step t as a column vector
            combined = np.vstack((h, xt))  # stack previous hidden state and current input

            fg = self.sigmoid(np.dot(self.W_f, combined) + self.b_f)  # forget gate
            ig = self.sigmoid(np.dot(self.W_i, combined) + self.b_i)  # input gate
            cg = np.tanh(np.dot(self.W_c, combined) + self.b_c)  # candidate cell state
            c = fg * c + cg * ig  # update cell state

            og = self.sigmoid(np.dot(self.W_o, combined) + self.b_o)  # output gate
            h = og * np.tanh(c)  # update hidden state

            self.cache.append((h, c, fg, ig, cg, combined))
            h_seq.append(h)

        return np.array(h_seq)
