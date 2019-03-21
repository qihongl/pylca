""" A leaky competing accumulator.
This can be viewed as a recurrrent neural network with lateral inhibition.
Functionally, it is related to DDM, but it works for n-choices.

Notes:
The implementation is based on [1][2][3].
TODO: Explain the differences here...

References:
[1] Usher, M., & McClelland, J. L. (2001).
The time course of perceptual choice: the leaky, competing accumulator model.
Psychological Review, 108(3), 550–592.
[2] Polyn, S. M., Norman, K. A., & Kahana, M. J. (2009).
A context maintenance and retrieval model of organizational processes in free
recall. Psychological Review, 116(1), 129–156.
[3] PsyNeuLink: https://github.com/PrincetonUniversity/PsyNeuLink
"""
import numpy as np


class LCA():

    def __init__(
        self, n_units, dt, leak, competition, self_excit=0,
        w_input=1, w_cross=0, offset=0, noise_sd=0
    ):
        # number of accumulators
        self.n_units = n_units
        # decision parameters
        self.leak = leak
        self.self_excit = self_excit
        self.competition = competition
        # the input-output weights
        self.w_input = w_input
        self.w_cross = w_cross
        # the "additive drift" terms at time t, like the bias
        self.offset = offset
        # noise on the accumulator
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
        assert 0 <= self.competition,\
            f'Invalid competition = {self.competition}'
        assert 0 <= self.self_excit, \
            f'Invalid self excitation = {self.self_excit}'
        assert 0 <= self.dt <= 1, f'Invalid dt = {self.dt}'
        assert 0 <= self.noise_sd, f'Invalid noise sd = {self.noise_sd}'

    def run(self, stimuli, threshold=1):
        """Run LCA on some stimulus sequence
        the update formula:
            1. value =   prev value
                      + new_input
                      - leaked previous value
                      + previous value updated with recurrent weigths
                      + offset and noise
            2. value <- output_bounding(value)

        Parameters
        ----------
        stimuli : 2d array
            input sequence, with shape: T x n_units
        threshold: float
            the upper bound of the neural activity

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
            scale=self.noise_sd, size=(self.n_units, len(stimuli)))
        # precompute the transformed input, for all time points
        inp = stimuli @ self.W_io
        # precompute offset
        offset = self.offset * np.ones(self.n_units,)
        # prealloc values for the accumulators over time
        V = np.zeros((self.n_units, len(stimuli)))
        # loop over n_cycles
        init_val = np.zeros((self.n_units,))
        for t in range(T):
            # the LCA computation at time t
            V_prev = init_val if t == 0 else V[:, t-1]
            V[:, t] = V_prev + offset + noise[:, t] * (self.dt**.5) + \
                (inp[t, :] - self.leak * V_prev + self.W_oo @ V_prev) * self.dt
            # output bounding
            V[:, t][V[:, t] < 0] = 0
            V[:, t][V[:, t] > threshold] = threshold
        return V.T


def make_weights(diag_val, offdiag_val, n_nodes):
    """Get a connection weight matrix with "diag-offdial structure"

    Parameters
    ----------
    diag_val : float
        the value of the diag entries
    offdiag_val : float
        the value of the off-diag entries
    n_nodes : int
        the number of LCA nodes

    Returns
    -------
    2d array
        the weight matrix with "diag-offdial structure"

    """
    # set up the masks
    diag_mask = np.matrix(np.eye(n_nodes))
    offdiag_mask = np.ones((n_nodes, n_nodes))
    np.fill_diagonal(offdiag_mask, 0)
    # compute the masks
    weight_matrix = diag_mask * diag_val + offdiag_mask * offdiag_val
    return weight_matrix
