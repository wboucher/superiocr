
import numpy as np

def tanh(z):
    #e2x = np.exp(-2 * z)
    #return (1 - e2x) / (1 + e2x)
    return np.tanh(z)

def tanh_grad(z):
    ex = np.exp(-1 * z)
    return (2 * ex / (1 + ex ** 2))**2

class Layer:
    def forward(self, input_vector):
        pass
    def back(self, error_vector):
        pass
    def grad():
        pass

class TanhLayer(Layer):
    def __init__(self, input_dim, output_dim):
        self.Phi = np.random.randn(input_dim, output_dim) / 100

    def forward(self, inputs):
        last = inputs[len(inputs) - 1]
        input_v = np.hstack((np.ones((len(last), 1)), last))
        return inputs + [tanh(np.dot(input_v, self.Phi))]
    
    def back(self, alpha, gamma, input_vector, error_vector):
        I = np.hstack((np.ones((len(input_vector), 1)), input_vector))

        tanh_value = input_vector
        tanh_grad  = 1 - np.multiply(tanh_value, tanh_value)

        di = np.multiply(np.dot(error_vector, self.Phi[1:, :].transpose()), tanh_grad)
        phi_grad = np.dot(I.transpose(), error_vector)

        self.Phi += alpha * (phi_grad - gamma * self.Phi)
        return di

class SigmoidLayer(Layer):
    def __init__(self, input_dim, output_dim):
        self.Phi = np.random.randn(input_dim, output_dim) / 100

    def forward(self, inputs):
        last = inputs[len(inputs) - 1]
        input_v = np.hstack((np.ones((len(last), 1)), last))
        return inputs + [tanh(np.dot(input_v, self.Phi))]
    
    def back(self, alpha, gamma, input_vector, error_vector):
        I = np.hstack((np.ones((len(input_vector), 1)), input_vector))

        in_value = input_vector
        tran_grad  = np.multiply(input_vector, 1 - input_vector)

        di = np.multiply(np.dot(error_vector, self.Phi[1:, :].transpose()), tran_grad)
        phi_grad = np.dot(I.transpose(), error_vector)

        self.Phi += alpha * (phi_grad - gamma * self.Phi)
        return di

class NNModel:
    def __init__(self, layers, max_label, alpha, gamma, steps = 1000):
        self.layers = layers
        self.alpha  = alpha
        self.gamma  = gamma
        self.steps  = steps
        self.max_label = max_label

    def forward(self, X):
        #print "forward in model"
        ai = [X]
        for layer in self.layers:
            ai = layer.forward(ai)
        return ai

    def backprop(self, X, target):
        #print "starting backprop"

        p = np.random.permutation(len(X))
        X = X[p]
        target = target[p]

        activations = self.forward(X)
        di_base = target - activations[len(activations) - 1]

        depth = len(self.layers)
        #print "backpropagating..."
        for n in range(len(X)):
            #di = di_base[n]
            activations = self.forward(X[n])
            di = target[n] - activations[len(activations) - 1]
            for i in reversed(range(depth)[1:]):
                #print i
                #print di[n].shape
                #print activations[i + 1][n].shape
                #print activations[i][n].shape
                #print self.layers[i].Phi.shape

                di = self.layers[i].back(self.alpha, self.gamma, activations[i], di)
                #di = self.layers[i].other_back(self.alpha, self.gamma, activations[i], activations[i+1], di)

    def predict(self, X, raw = False):
        final = self.forward(X)
        final = final[len(final) - 1]
        if raw:
            return final
        else:
            return final.argmax(axis = 1)

    def train(self, X, y, steps=None):
        def to_label(b):
            if b:
                return 1
            else:
                return -1
        T = np.array([[to_label(v == c + 1) for c in range(self.max_label)] for v in y])

        if not steps:
            steps = self.steps

        cent = steps / 100
        for i in range(steps):
            if i % cent == 0:
                print i / cent
            self.backprop(X, T)
