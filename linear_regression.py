from matplotlib import pyplot, pylab
from numpy.linalg import inv
from numpy import linalg as LA
import numpy as np
import random


class LinnearReggresion:
    def __init__(self, points=2, vector_size=2):
        self.missmatch = []
        self.points = points
        self.vector_size = vector_size

        self.weights = np.zeros(self.vector_size)
        # generates vector
        self.features = np.array([self.generate_data()
                                  for _ in range(self.points)])

    def generate_data(self):
        '''
            Its obvious. 
            Generate feature and labels.
        '''
        features = np.array([random.uniform(-1, 1)
                             for _ in range(self.vector_size)])
        label = int(np.sign(sum(features)))
        return np.array([features, label])

    def train(self):
        '''
            Train the algoritm, compute pseudo-invr
            matrix.
        '''
        x = []
        y = []

        for feature, label in self.features:
            x.append(feature)
            y.append(label)

        #algorithm
        X = np.matrix(x)
        Y = np.array(y)
        Xt = X.transpose()
        Xd = inv(Xt.dot(X)).dot(Xt)
        self.weights += np.array(Xd).dot(Y)

    def define_missmatch(self):
        '''
            Define missmatch by disagreement aka error.
        '''
        n = LA.norm(self.vector_size)  # aka the length of p.w vector
        #ww = self.weights / n  # a unit vector
        missmatch = 0
        for feature, label in self.features:
            if int(np.sign(np.dot(self.weights, feature))) != label:
                missmatch += 1
        result = missmatch / float(len(self.features))
        self.missmatch.append(result)

        print self.weights
        print self.features
        print sum(self.missmatch) / len(self.missmatch)

    def draw_plot(self):
        '''
            Outputs foo.png result image.
        '''
        n = LA.norm(self.vector_size) # aka the length of p.w vector
        ww = self.weights / n # a unit vector
        for feature, label in self.features:
            if label == 1:
                pyplot.plot(feature[0], feature[1], 'ob')
            else:
                pyplot.plot(feature[0], feature[1], 'or')

        pylab.ylim([-1, 1])
        pylab.xlim([-1, 1])

        ww1 = [ww.item(1), -ww.item(0)]
        ww2 = [-ww.item(1), ww.item(0)]
        pyplot.plot([ww1[0], ww2[0]], [ww1[1], ww2[1]], 'k')
        pyplot.savefig('foo.png')

    '''
    Meh, you can use it later.

        def generate_data(self):
            features = np.array([random.uniform(-1, 1)
            for _ in range(self.vector_size )])
            label = int(np.sign(sum(features)))
            #label = np.array(random.choice([-1, 1]))
            return np.array([features, label])
    '''

if __name__ == "__main__":
    lr = LinnearReggresion(10)
    lr.train()
    lr.define_missmatch()
