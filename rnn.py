import numpy as np
from chainer import Variable
from chainer import optimizers
from chainer import Chain
import chainer.links as L
import chainer.function as F
from chainer.functions.evaluation.accuracy import accuracy
from chainer.functions.loss.sigmoid_cross_entropy import sigmoid_cross_entropy
from chainer.functions.loss.softmax_cross_entropy import softmax_cross_entropy


class rnn(Chain):
    def __init__(self, n_units, n_out):
        super(rnn, self).__init__(
            l0=L.Linear(n_units, n_units * 2),
            l1=L.Linear(n_units * 2, n_units * 4),  # n_in -> n_units
            l2=L.Linear(n_units * 4, n_units * 4),  # n_units -> n_units
            l3=L.LSTM(n_units * 4, n_out),  # n_units -> n_out
        )

    def __call__(self, x):
        h = self.l0(x)
        h2 = self.l1(h)
        h3 = self.l2(h2)
        h4 = self.l3(h3)
        return h4

    def forward(self, x):
        h = self.l0(x)
        h2 = self.l1(h)
        h3 = self.l2(h2)
        h4 = self.l3(h3)
        return h4

    def reset_state(self):
        self.l3.reset_state()


class trainer:
    def __init__(self, n_units, n_out):
        self.model = rnn(n_units, n_out)
        self.optimizer = optimizers.SGD(lr=10)
        self.optimizer.setup(self.model)

    def run(self):
        listt = np.asarray([[1, 28, 3, 41], [10, 2, 30, 4], [71, 21, 3, 46], [11, 27, 3, 41]]).astype(np.float32)
        modelclass = L.Classifier(self.model)
        epochs = 1000
        for i in range(epochs):
            for x in listt:
                self.model.reset_state()
                xset = Variable(np.asarray(x).astype(np.float32).reshape(1, -1))
                self.optimizer.zero_grads()

                if x[0] == 1:
                    to = Variable(np.asarray([1, 0]).reshape(1,-1))
                if x[0] == 10:
                    to = Variable(np.asarray([2, 1]).reshape(1, -1))
                if x[0] == 71:
                    to = Variable(np.asarray([3, 0]).reshape(1, -1))
                if x[0] == 11:
                    to = Variable(np.asarray([4, 1]).reshape(1, -1))

                r = modelclass(xset, to)
                # print('\n', r.data)

                racc = Variable(np.asarray([r.data]).reshape(1, 1))
                xacc = Variable(np.asarray(x).astype(np.int32).reshape(1, -1))
                acct = accuracy(racc, to)

                rloss = Variable(np.asarray(r.data).reshape(2, -1))
                tloss = Variable(np.asarray(to.data).reshape(2, -1))
                loss = softmax_cross_entropy(rloss, tloss)
                loss.backward()
                print(acct.data, loss.data)
                self.optimizer.update()


nntrainer = trainer(4, 2)
nntrainer.run()
