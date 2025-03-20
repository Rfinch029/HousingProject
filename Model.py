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
    def forward(self, x):  # here, x is a 2D numpy array. For us, this would be the weather parameters at time t.
        T = x.shape[0]  # T represents length of input
        h, c = np.zeros((self.hidden_dim, 1))  # h (hidden state) and c (cell state) are initialized as empty vectors. Will be updated

        self.cache = [] # stores intermediate values (that will be added later), required for back-propagation
        h_seq = [] # stores hidden_states (short-term memory) as it is being updated.

        for t in range(T):
            xt = x[t].reshape(-1,1)  # for the number of parameters, reshape each parameter as a column vector
            combined = np.vstack((h, xt))  # stacks hidden state and input into a combined "column vector"

            fg = self.sigmoid(np.dot(self.W_f, combined) + self.b_f)  # calculations for forget gate
            # not sure why activation functions is giving error
            ig = self.sigmoid(np.dot(self.W_i, combined) + self.b_i)  # calculations for input gate.
            cg = np.tanh(np.dot(self.W_c, combined) + self.b_c)  # calculations for a potential new cell
            c = fg * c + cg * ig  # updating cell state value

            og = self.sigmoid(np.dot(self.W_o, combined) + self.b_o)  # calculating for output gate
            h = og * np.tanh(c)

            self.cache.append((h, c, fg, ig, cg, combined))  # adding these updated values to the cache. Each tuple represents one time step.
            h_seq.append(h)  # updating hidden state value

        return np.array(h_seq)  # returns the vector h_seq for the next iteration as a nparray.

    # CONFUSING: dL_dh_seq stands for derivative of Loss with respect to hidden state. We are inputting a list of gradient losses
    def backward(self, dL_dh_seq, learning_rate = 0.01):
        # These gradient accumulators will store the sum of gradients across all time steps
        dW_f = np.zeros_like(self.W_f)
        db_f = np.zeros_like(self.b_f)
        dW_i = np.zeros_like(self.W_i)
        db_i = np.zeros_like(self.b_i)
        dW_c = np.zeros_like(self.W_c)
        db_c = np.zeros_like(self.b_c)
        dW_o = np.zeros_like(self.W_o)
        db_o = np.zeros_like(self.b_o)

        # These next step gradients are propagated from the future time step to the current one.
        d_next_h = np.zeros((self.hidden_dim, 1))
        d_next_c = np.zeros((self.hidden_dim, 1))

        # backpropagation through time
        for t in reversed(range(len(self.cache))):  # we are iterating backwards through self.cache (intermediate values) because of chain rule.
            (h, c, fg, ig, cg, combined) = self.cache[t]  # accessing various values from cache.
            dL_dh = dL_dh_seq[t] + d_next_h
            d_ot = dL_dh * np.tanh(c) * ot * (1 - ot)
            dL_dc = dL_dh * ot * (1 - np.tanh(c) ** 2) + d_next_c

            d_ft = dL_dc * c * ft * (1 - ft)
            d_it = dL_dc * c_tilde * it * (1 - it)
            d_c_tilde = dL_dc * it * (1 - c_tilde ** 2)

            dW_f += np.dot(d_ft, combined.T)
            db_f += d_ft
            dW_i += np.dot(d_it, combined.T)
            db_i += d_it
            dW_c += np.dot(d_c_tilde, combined.T)
            db_c += d_c_tilde
            dW_o += np.dot(d_ot, combined.T)
            db_o += d_ot

            d_combined = (np.dot(self.W_f.T, d_ft) +
                          np.dot(self.W_i.T, d_it) +
                          np.dot(self.W_c.T, d_c_tilde) +
                          np.dot(self.W_o.T, d_ot))

            d_next_h = d_combined[:self.hidden_dim, :]
            d_next_c = dL_dc * ft

            # Update weights using gradient descent
        self.W_f -= learning_rate * dW_f
        self.b_f -= learning_rate * db_f
        self.W_i -= learning_rate * dW_i
        self.b_i -= learning_rate * db_i
        self.W_c -= learning_rate * dW_c
        self.b_c -= learning_rate * db_c
        self.W_o -= learning_rate * dW_o
        self.b_o -= learning_rate * db_o




