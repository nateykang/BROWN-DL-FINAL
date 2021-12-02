
import tensorflow as tf
import numpy as np
from preprocessing import get_data

class Model(tf.keras.Model):
    def __init__(self):
        """
        NEED TO EDIT THIS
        """
        super(Model, self).__init__()

        self.batch_size = 100
        self.num_teams = 127
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
    #train_data = get_data(r"C:\Users\natha\OneDrive\Documents\GitHub\cs1470\BROWN-DL-FINAL-EM_NK\data\recruiting_rankings_2014.csv",
    #r"C:\Users\natha\OneDrive\Documents\GitHub\cs1470\BROWN-DL-FINAL-EM_NK\data\recruiting_rankings_2015.csv",
    #r"C:\Users\natha\OneDrive\Documents\GitHub\cs1470\BROWN-DL-FINAL-EM_NK\data\recruiting_rankings_2016.csv",
    #r"C:\Users\natha\OneDrive\Documents\GitHub\cs1470\BROWN-DL-FINAL-EM_NK\data\recruiting_rankings_2017.csv",
    #r"C:\Users\natha\OneDrive\Documents\GitHub\cs1470\BROWN-DL-FINAL-EM_NK\data\team_records_2014.csv",
    #r"C:\Users\natha\OneDrive\Documents\GitHub\cs1470\BROWN-DL-FINAL-EM_NK\data\team_records_2015.csv",
    #r"C:\Users\natha\OneDrive\Documents\GitHub\cs1470\BROWN-DL-FINAL-EM_NK\data\team_records_2016.csv",
    #r"C:\Users\natha\OneDrive\Documents\GitHub\cs1470\BROWN-DL-FINAL-EM_NK\data\team_records_2017.csv",
    #r"C:\Users\natha\OneDrive\Documents\GitHub\cs1470\BROWN-DL-FINAL-EM_NK\data\returning_production_2018.csv",)

    train_data = get_data(
        'data/recruiting_rankings_2014.csv',
        'data/recruiting_rankings_2015.csv',
        'data/recruiting_rankings_2016.csv',
        'data/recruiting_rankings_2017.csv',
        'data/team_records_2014.csv',
        'data/team_records_2015.csv',
        'data/team_records_2016.csv',
        'data/team_records_2017.csv',
        'data/returning_production_2018.csv')


if __name__ == '__main__':
    main()
