import pandas as pd
import numpy as np
import os

current_path = os.getcwd()
input_path = current_path + '/data/Input/'
output_path = current_path + '/data/ML/Results/'

for i in range(1):
    # lines = pd.read_csv(input_path + 'Input' + str(i + 1) + ".txt", delimiter='\t', header=None).values
    # X = lines

    output = pd.read_csv(output_path + 'Commitment' + str(i + 1) + '.txt', delimiter=' ', header=None).values[0]
    output_reshape = np.reshape(output[0:-1], (24, 54))
    pd.DataFrame(output_reshape).to_csv(output_path + 'output1.csv')
    commitment = np.where(output_reshape > 0.5, 1, -1)
    y = commitment

unit = np.zeros((998,24,54), dtype=np.int32)
for i in range(998):
    # lines = pd.read_csv(input_path + 'Input' + str(i + 2) + ".txt", delimiter='\t', header=None).values
    # X = np.vstack([X, lines])
    output = pd.read_csv(output_path + 'Commitment' + str(i + 2) + '.txt', delimiter=' ', header=None).values[0]
    output_reshape = np.reshape(output[0:-1], (24, 54))
    commitment = np.where(output_reshape > 0.5, 1, -1)
    unit[i] = commitment
print(unit.shape)

# i = 0
# count_uniq_hour = 0
# for gen in range(54):
#     for hour in range(24):
#         while i < 998 and y[hour, gen] == unit[i, hour, gen]:
#              i += 1
#         if i == 998:
#             count_uniq_hour += 1
#         i = 0
# print(count_uniq_hour)

i = 0
count_uniq_day = 0
for gen in range(54):
    while i < 998 and np.array_equal(y[:, gen], unit[i, :, gen]):
        i += 1
    if i == 998:
        count_uniq_day += 1
    i = 0
print(count_uniq_day)

