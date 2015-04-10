# Define the model used in the paper
import numpy as np
import matplotlib.pyplot as plt


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
        b_vec = np.reshape(self.b, (self.b.shape[0], self.b.shape[1]*self.b.shape[2]))
        b_t_vec = np.reshape(b_t, (b_t.shape[0], b_t.shape[1]*b_t.shape[2]))
        # outer refers to the product column by row which creates a rank-1
        # matrix out of 2 vectors
        logC = sum([yw[k] * np.outer(b_vec[k, :], b_t_vec[k, :])
                    for k in range(yw.shape[0])])
        print("logC :", logC)

        #We need to unroll the images
        x_vec = np.reshape(x, (x.shape[0]*x.shape[1]))
        b = np.dot(np.transpose(x_vec), np.dot(np.exp(-logC), x_vec))

        # Compute the prior on sparsity
        c = -sum(np.abs(y))

        return -0.5 * (a + b + c)

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


if __name__ == "__main__":
    n_features = 2
    s_features = 5
    w = np.random.rand(1, n_features)
    b = np.random.rand(n_features, s_features, s_features)
    mod = Model(w, b)
    #mod.show_features()
    x = np.random.rand(s_features, s_features)
    print(x)
    a = mod.forward_prop(x)
    print(a)
    print(mod.log_likelihood(x, a))
