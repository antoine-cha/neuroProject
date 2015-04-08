# Define the model used in the paper
import numpy as np


class Model:
    def __init__(self, w, b):
        """
        Defining the parameters of the model : weights and feature vectors
        WARNING : the features vectors are assumed to be of norm 1
        let a be the vector of activations after conv(b, x)
        y = w * a
        ------------------------------------------------------------------
        w : n_y * n_features matrix,
            weights the activations for each neurons
        b : 3D-tensors n_features * 20 * 20,
            consists in all the feature vectors
        """
        self.w = w  # weights of the connections
        self.b = b  # features vectors (filters)

    def forward_prop(self, x):
        """
        Given an input image, perform the forward propagation
        -----------------------------------------------------
        x : 2D matrix,
            input image, should be the same size as w
        """
        return np.dot(self.w, self.b*x)

    def log_likelihood(self, x, y):
        """
        Define using [S10] in the supplementary material
        -----------------------------------------------
        x : input image, considered as a vector /!\
        y : MAP of activations given x
        """
        yw = np.dot(np.transpose(y), self.w)  # Y^T * W
        a = np.sum(yw)
        # equation [S4] in supplementary material
        logC = sum([yw[k] * np.dot(self.b[k, :, :], np.transpose(self.b[k, :, :]))
                    for k in range(yw.shape[0])])
        # x^T * exp(-log C) * x
        b = np.dot(np.transpose(x), np.dot(np.exp(-logC), x))
        # Compute the prior on sparsity
        c = -sum(np.abs(y))

        return -0.5 * (a + b + c)
