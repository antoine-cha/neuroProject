# Define the model used in the paper
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


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
        if(w.shape[1] != b.shape[0]):
            raise ValueError(" Dimensions should match \n" +
                             " w.shape[1]!=b.shape[0]")

        self.w = w/np.sum(w)  # weights of the connections
        self.b = b  # features vectors (filters)

    def forward_prop(self, x):
        """
        Given an input image, perform the forward propagation
        -----------------------------------------------------
        x : 2D matrix,
            input image, should be the same size as w
        """
        c = np.sum(self.b*x, axis=(1, 2))
        print(c.shape)
        print(self.w.shape)
        return np.dot(self.w, c)

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
        b_t = np.transpose(self.b, (0, 2, 1))
        b_vec = np.reshape(self.b,
                           (self.b.shape[0], self.b.shape[1]*self.b.shape[2]))
        b_t_vec = np.reshape(b_t, (b_t.shape[0], b_t.shape[1]*b_t.shape[2]))
        # outer refers to the product column by row which creates a rank-1
        # matrix out of 2 vectors
        logC = sum([yw[k] * np.outer(b_vec[k, :], b_t_vec[k, :])
                    for k in range(yw.shape[0])])

        # We need to unroll the images
        x_vec = np.reshape(x, (x.shape[0]*x.shape[1]))
        # Use expm to get the exponential of the matrix
        # and not the matrix of the exp of the components
        b = np.dot(np.transpose(x_vec), np.dot(expm(-logC), x_vec))

        return -0.5 * (a + b)

    def grad_b_ll(self, x, y, k):
        """
        Compute the gradient of the loglikelihood in bk
        -----------------------------------------------
        """
        b_t = np.transpose(self.b, (0, 2, 1))
        x_vec = np.reshape(x, (x.shape[0]*x.shape[1]))
        b_vec = np.reshape(self.b,
                           (self.b.shape[0], self.b.shape[1]*self.b.shape[2]))
        b_t_vec = np.reshape(b_t, (b_t.shape[0], b_t.shape[1]*b_t.shape[2]))
        yw = np.dot(np.transpose(y), self.w)  # Y^T * W
        logC = sum([yw[k] * np.outer(b_vec[k, :], b_t_vec[k, :])
                    for k in range(yw.shape[0])])
        res = np.dot(np.transpose(x_vec), expm(-logC))
        res_ = np.sum(yw)*np.outer(b_vec[k, :], x_vec)
        return np.dot(res, res_)

    def show_features(self):
        """
        Display the feature vectors as black and white images
        """
        n = self.b.shape[0]
        p = n//10 + 1
        if (n % 10 == 0):
            p -= 1
        for i in range(n):
            plt.subplot(p, 10, i)
            plt.axis('off')
            plt.imshow(b[i, :, :], cmap='gray')
        plt.show()
        return 0

    def map(x, y0, w, b, eta):
        """
        Calculate the MAP y
        -----------------------------------------------
        x : n_features * 1,
            input image, considered as a vector /!\
        y0 : n_features * 1,
            initial guess for y
        w : n_y * n_features matrix,
            weights the activations for each neurons
        b : n_features * 20^2,
            consists in all the feature vectors
        eta : step for the gradient descent
        """

        y = y0
        bx = np.dot(b, x) - 1
        wbx = np.dot(w, bx)

        while True: #/!\ Modify this line /!\
            p = np.abs(p)
            p = np.prod(np.exp(-p))
            y = y + eta*(wbx + p)

        return y


if __name__ == "__main__":
    n_features = 2
    s_features = 5
    w = np.random.rand(1, n_features)
    b = np.random.rand(n_features, s_features, s_features)
    b = b - np.mean(b)
    mod = Model(w, b)
    x = np.random.rand(s_features, s_features)
    print("test patch :")
    print(x)
    a = mod.forward_prop(x)
    print(a)
    print(mod.log_likelihood(x, a))
    print(mod.grad_b_ll(x, a, 1))
