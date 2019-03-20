"""
The leaky, competing, accumulator class

Reference:
Usher, M., & McClelland, J. L. (2001).
The time course of perceptual choice: the leaky, competing accumulator model.
Psychological Review, 108(3), 550–592.
Retrieved from https://www.ncbi.nlm.nih.gov/pubmed/11488378
"""
import numpy as np


class LCA():

    def __init__(
            self,
            n_units,
            leak, self_excit, competition,
            w_input, w_cross,
            offset, bias, gain,
            dt, noise_mu, noise_sd
    ):
        # number of accumulators
        self.n_units = n_units
        # decision parameters
        self.leak = leak
        self.self_excit = self_excit
        self.competition = competition
        # the accumulator output transfermation
        self.transfer_func = sigmoid
        # the input-output weights
        self.w_input = w_input
        self.w_cross = w_cross
        # the "shift" terms at time t, like the bias term
        self.offset = offset
        # the "intercept" and "slope" for the logistic function
        self.bias = bias
        self.gain = gain
        # noise on the accumulator
        self.noise_mu = noise_mu
        self.noise_sd = noise_sd
        # time step size
        self.dt = dt
        # the input-output weights
        self.W_io = make_weights(w_input, w_cross, n_units)
        # the recurrent weights on the output units (i.e. the accumulators)
        self.W_oo = make_weights(self_excit, -competition, n_units)
        # check params
        self.check_config()

    def check_config(self):
        assert 0 <= self.leak <= 1, f'Invalid leak = {self.leak}'
        assert 0 <= self.competition, f'Invalid competition = {self.competition}'
        assert 0 <= self.dt, f'Invalid dt = {self.dt}'
        assert 0 <= self.noise_sd, f'Invalid noise sd = {self.noise_sd}'

    def run(self, stimuli):
        """Run LCA on some stimulus sequence
        the update formula:
            value =   prev value
                      + new_input
                      - leaked previous value
                      + previous value updated with recurrent weigths
                      + offset and noise

        Parameters
        ----------
        stimuli : 2d array
            input sequence, with shape: T x n_units

        Returns
        -------
        2d array
            LCA acitivity time course, with shape = np.shape(stimuli)

        """
        # input validation
        T, n_units_ = np.shape(stimuli)
        assert n_units_ == self.n_units
        # precompute noise, for all time points
        noise = np.random.normal(
            loc=self.noise_mu, scale=self.noise_sd, size=(
                self.n_units, len(stimuli)))
        # precompute the transformed input, for all time points
        inp = stimuli @ self.W_io
        # precompute offset
        offset = self.offset * np.ones(self.n_units,)
        # prealloc values for the accumulators over time
        V = np.zeros((self.n_units, len(stimuli)))
        # loop over time
        V[:, 0] = self.transfer_func(
            np.zeros((self.n_units,)), self.bias, self.gain)
        for t in np.arange(1, T):
            # the LCA computation at time t
            V[:, t] = V[:, t-1] + offset + noise[:, t] * (self.dt**.5) + \
                (inp[t, :] - self.leak * V[:, t-1] + self.W_oo @ V[:, t-1])
            # transform
            V[:, t] = self.transfer_func(V[:, t], self.bias, self.gain)
        return V.T


# helper funcs
def make_weights(w_diag_val, w_offdiag_val, n_nodes):
    """Get a connection weight matrix with "diag-offdial structure"

    Parameters
    ----------
    w_diag_val : float
        the value of the diag entries
    w_offdiag_val : float
        the value of the off-diag entries
    n_nodes : int
        the number of LCA nodes

    Returns
    -------
    2d array
        the weight matrix with "diag-offdial structure"

    """
    diag_mask = np.matrix(np.eye(n_nodes))
    offdiag_mask = np.ones((n_nodes, n_nodes))
    np.fill_diagonal(offdiag_mask, 0)
    input_weights = diag_mask * w_diag_val + offdiag_mask * w_offdiag_val
    return input_weights


def sigmoid(x, bias=0, gain=1):
    return 1 / (1 + np.exp(-gain * (x+bias)))
