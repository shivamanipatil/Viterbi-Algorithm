from __future__ import division

import math


class Viterbi():
    """
    Viterbi
    https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, O, S, Y, A, B):

        self.O = O  # Observation space
        self.S = S  # State space
        self.Y = Y  # Sequence of observations
        self.A = A  # Transition matrix
        self.B = B  # Emission matrix

        self.N = len(self.O)
        self.K = len(self.S)

        # Word index lookup table
        self.lookup = {}
        for i, word in enumerate(self.O):
            self.lookup[word] = i

        self.T = len(Y)
        self.T1 = [[0] * self.T for i in range(self.K)]
        self.T2 = [[None] * self.T for i in range(self.K)]

        # Predicted tags
        self.X = [None] * self.T

    def decode(self):
        """
        Run the algorithm
        """
        # Initialize start probabilities
        self.init()
        # Forward step
        self.forward()
        # Backward step
        self.backward()
        return self.X
