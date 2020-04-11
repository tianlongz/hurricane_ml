import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


current_path = os.getcwd()
input_path = current_path + '/data/Input/'
output_path = current_path + '/data/ML/Results/'

for i in range(1):
    lines = pd.read_csv(input_path + 'Input' + str(i + 1) + ".txt", delimiter='\t', header=None).values
    X = lines

    output = pd.read_csv(output_path + 'Commitment' + str(i + 1) + '.txt', delimiter=' ', header=None).values[0]
    output_reshape = np.reshape(output[0:-1], (24, 54))
    pd.DataFrame(output_reshape).to_csv(output_path + 'output1.csv')
    commitment = np.where(output_reshape > 0.5, 1, -1)
    y = commitment
    print(y)

for i in range(998):
    lines = pd.read_csv(input_path + 'Input' + str(i + 2) + ".txt", delimiter='\t', header=None).values
    X = np.vstack([X, lines])

    output = pd.read_csv(output_path + 'Commitment' + str(i + 2) + '.txt', delimiter=' ', header=None).values[0]
    output_reshape = np.reshape(output[0:-1], (24, 54))
    commitment = np.where(output_reshape > 0.5, 1, -1)
    y = np.vstack([y, commitment])

    # print(lines.shape)

"""
Create dataset of classification task with many redundant and few informative features
"""

for commitment_number in range(54):
    Y = y[:, commitment_number]

    if np.unique(Y).shape[0] != 2:
        continue

    # print(Y.shape)
    print("commitment number: ", commitment_number+1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    # print(X_train.shape)
    # print(y_train.shape)
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)

    y_pred = svclassifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# -----------------------------------------------------------
# see which file is not in there (output file #985 is missing)

# for i in range(1000):
#     try:
#         f = open(output_path + 'Commitment' + str(i + 1) + ".txt")
#     except IOError:
#         print(i+1, "File not accessible")
#     finally:
#         f.close()
# -----------------------------------------------------------

