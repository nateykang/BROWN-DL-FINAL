
import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def __init__(self):
        """
        NEED TO EDIT THIS
        """
        super(Model, self).__init__()

        self.batch_size = 100
        self.num_teams = # Need to fill this in
        self.learning_rate = tf.keras.optimizers.Adam(.001)


    def call(self, inputs):
        """
        NEED TO FILL IN
        """

        pass

    def loss(self, logits, labels):
        """
        NEED TO FILL IN
        """
        pass

    def accuracy(self, logits, labels):
        """
        NEED TO FILL IN
        """
        pass


def train(model, train_inputs, train_labels):
    '''
    NEED TO FILL IN
    '''

    pass


def test(model, test_inputs, test_labels):
    """
    NEED TO FILL IN
    """
    pass


def main():
    '''
    NEED TO FILL IN
    '''

    pass


if __name__ == '__main__':
    main()
