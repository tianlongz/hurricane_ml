import pandas as pd
import scipy as sci
import numpy as np
import os
import csv

current_path = os.getcwd()
input_path = current_path + '/data/Input/'
output_path = current_path + '/data/ML/Results/'


for i in range(1):
    input = pd.read_csv(input_path + 'Input' + str(i + 1) + ".txt", delimiter='\t', header=None).values

    output = pd.read_csv(output_path + 'Commitment' + str(i + 1) + '.txt', delimiter=' ', header=None).values[0]
    output = output[0:-1]
    output = np.reshape(output, (24, 54))
    print(output.shape)
    print(output[0])


