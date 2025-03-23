import numpy as np

class LSTM:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Xavier Initialization for weights and zeros for biases.
        self.W_f = np.random.randn(hidden_dim, input_dim + hidden_dim) * np.sqrt(1 / (input_dim + hidden_dim))
        self.b_f = np.zeros((hidden_dim, 1))

        self.W_i = np.random.randn(hidden_dim, input_dim + hidden_dim) * np.sqrt(1 / (input_dim + hidden_dim))
        self.b_i = np.zeros((hidden_dim, 1))

        self.W_c = np.random.randn(hidden_dim, input_dim + hidden_dim) * np.sqrt(1 / (input_dim + hidden_dim))
        self.b_c = np.zeros((hidden_dim, 1))

        self.W_o = np.random.randn(hidden_dim, input_dim + hidden_dim) * np.sqrt(1 / (input_dim + hidden_dim))
        self.b_o = np.zeros((hidden_dim, 1))

    @staticmethod
    def sigmoid(x):
        # Clip x to avoid overflow issues in the exponential
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2

    def forward(self, x):
        T = x.shape[0]
        h = np.zeros((self.hidden_dim, 1))
        c = np.zeros((self.hidden_dim, 1))
        self.cache = []  # Clear any previous cache.
        h_seq = []
        for t in range(T):
            xt = x[t].reshape(-1, 1)
            combined = np.vstack((h, xt))
            fg = self.sigmoid(np.dot(self.W_f, combined) + self.b_f)
            ig = self.sigmoid(np.dot(self.W_i, combined) + self.b_i)
            cg = np.tanh(np.dot(self.W_c, combined) + self.b_c)
            c = fg * c + ig * cg
            og = self.sigmoid(np.dot(self.W_o, combined) + self.b_o)
            h = og * np.tanh(c)
            self.cache.append((h, c, fg, ig, cg, og, combined))
            h_seq.append(h)
        return np.array(h_seq)

    def step_forward(self, h, c, x_next):
        """
        Runs a single LSTM forward step given the previous hidden state (h),
        previous cell state (c), and the new input x_next.

        Args:
            h (np.ndarray): Previous hidden state of shape (hidden_dim, 1)
            c (np.ndarray): Previous cell state of shape (hidden_dim, 1)
            x_next (np.ndarray): New input features of shape (input_dim,) or (input_dim, 1)

        Returns:
            h_next (np.ndarray): Updated hidden state (shape: (hidden_dim, 1))
            c_next (np.ndarray): Updated cell state (shape: (hidden_dim, 1))
        """
        if x_next.ndim == 1:
            x_next = x_next.reshape(-1, 1)  # Ensure x_next is a column vector

        combined = np.vstack((h, x_next))

        fg = self.sigmoid(np.dot(self.W_f, combined) + self.b_f)
        ig = self.sigmoid(np.dot(self.W_i, combined) + self.b_i)
        cg = np.tanh(np.dot(self.W_c, combined) + self.b_c)
        c_next = fg * c + ig * cg
        og = self.sigmoid(np.dot(self.W_o, combined) + self.b_o)
        h_next = og * np.tanh(c_next)

        return h_next, c_next

    def backward(self, dh_seq):
        """
        Performs a backward pass through time (BPTT) given the gradients with respect
        to the hidden states (dh_seq) from the output layer.
        dh_seq: gradient of loss w.r.t. each hidden state; shape (T, hidden_dim, 1)

        Returns:
            A dictionary of gradients for the LSTM parameters.
        """
        T = len(self.cache)
        dW_f = np.zeros_like(self.W_f)
        db_f = np.zeros_like(self.b_f)
        dW_i = np.zeros_like(self.W_i)
        db_i = np.zeros_like(self.b_i)
        dW_c = np.zeros_like(self.W_c)
        db_c = np.zeros_like(self.b_c)
        dW_o = np.zeros_like(self.W_o)
        db_o = np.zeros_like(self.b_o)

        dh_next = np.zeros((self.hidden_dim, 1))
        dc_next = np.zeros((self.hidden_dim, 1))

        for t in reversed(range(T)):
            h, c, fg, ig, cg, og, combined = self.cache[t]
            # Total gradient for this time step's hidden state:
            dh = dh_seq[t] + dh_next

            # Gradients for output gate:
            do = dh * np.tanh(c)
            d_og = do * og * (1 - og)

            # Gradient for cell state:
            dtanh_c = dh * og * (1 - np.tanh(c) ** 2)
            dc = dtanh_c + dc_next

            # Gradients for input gate, candidate, and forget gate:
            d_ig = dc * cg
            d_ig *= ig * (1 - ig)

            d_cg = dc * ig
            d_cg *= (1 - cg ** 2)

            # For forget gate, we need previous cell state.
            c_prev = self.cache[t - 1][1] if t > 0 else np.zeros_like(c)
            d_fg = dc * c_prev
            d_fg *= fg * (1 - fg)

            # Accumulate parameter gradients.
            dW_o += np.dot(d_og, combined.T)
            db_o += d_og

            dW_i += np.dot(d_ig, combined.T)
            db_i += d_ig

            dW_c += np.dot(d_cg, combined.T)
            db_c += d_cg

            dW_f += np.dot(d_fg, combined.T)
            db_f += d_fg

            # Backpropagate into the combined input.
            dcombined = (np.dot(self.W_o.T, d_og) +
                         np.dot(self.W_i.T, d_ig) +
                         np.dot(self.W_c.T, d_cg) +
                         np.dot(self.W_f.T, d_fg))
            # The first part of dcombined corresponds to dh from the previous time step.
            dh_prev = dcombined[:self.hidden_dim, :]
            # dx = dcombined[self.hidden_dim:, :]  # Not used here.

            dh_next = dh_prev
            dc_next = dc * fg

        grads = {
            'W_f': dW_f, 'b_f': db_f,
            'W_i': dW_i, 'b_i': db_i,
            'W_c': dW_c, 'b_c': db_c,
            'W_o': dW_o, 'b_o': db_o
        }
        return grads
