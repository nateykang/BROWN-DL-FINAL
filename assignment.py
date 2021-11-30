#assignment

import csv
import numpy as np
import tensorflow as tf

with open('ex_file.csv','r') as ex_file:
    data = csv.reader(ex_file, delimiter = ',')
data_array = np.asarray(data)