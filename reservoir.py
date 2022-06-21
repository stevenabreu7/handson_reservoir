import numpy as np


class Reservoir:
    def __init__(self, n_inputs, n_neurons, rhow=1.25, inp_scaling=1., leak_range=(0.1,0.3),
                 verbose=False):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.rhow = rhow
        self.inp_scaling = inp_scaling
        self.leak_range = leak_range

        # initialize weight matrices
        self.win = np.random.uniform(low=-1., high=1., size=(n_neurons, n_inputs+1)) * inp_scaling
        self.w = np.random.random((n_neurons, n_neurons)) * 2. - 1.
        leak_low, leak_high = leak_range
        self.leak = np.random.uniform(low=leak_low, high=leak_high, size=(n_neurons,))

        # set spectral radius
        rhow_current = self.spectral_radius
        self.w = self.w * rhow / rhow_current
        if verbose:
            print(f'spectral radius: {self.spectral_radius:.3f}')


    @property
    def spectral_radius(self):
        # compute the spectral radius
        return max(abs(np.linalg.eig(self.w)[0]))


    def forward(self, u, wout=None, collect_states=False):
        n_timesteps = u.shape[0]
        # initialize state
        x = np.zeros((self.n_neurons,))
        # setup matrix for states
        if collect_states or wout is not None:
            X = np.zeros((n_timesteps, self.n_neurons))
        # forward pass loop
        for t in range(n_timesteps):
            ut = u[t,:]
            x_next = np.tanh(self.win @ np.concatenate((ut, [1])) + self.w @ x)
            x = (1. - self.leak) * x + self.leak * x_next
            if collect_states or wout is not None:
                X[t,:] = x
        # compute outputs if desired
        if wout is not None:
            Y = X @ wout
        # return outputs and/or states
        if wout is not None and collect_states:
            return X, Y
        elif wout is not None:
            return Y
        elif collect_states:
            return X
