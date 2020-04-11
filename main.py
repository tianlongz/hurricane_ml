import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

current_path = os.getcwd()
input_path = current_path + '/data/Input/'
output_path = current_path + '/data/ML/Results/'
result_path = current_path + '/data/pred_results/'
line_path = current_path + '/data/Lines/'

for i in range(1):
    lines = pd.read_csv(input_path + 'Input' + str(i + 1) + ".txt", delimiter='\t', header=None).values
    X = np.reshape(lines, (1, 4464))

    output = pd.read_csv(output_path + 'Commitment' + str(i + 1) + '.txt', delimiter=' ', header=None).values[0]
    output_reshape = np.reshape(output[0:-1], (1, 1296))
    commitment = np.where(output_reshape > 0.5, 1, -1)
    y = commitment

for i in range(998):
    lines = pd.read_csv(input_path + 'Input' + str(i + 2) + ".txt", delimiter='\t', header=None).values
    X = np.vstack([X, np.reshape(lines, (1, 4464))])

    output = pd.read_csv(output_path + 'Commitment' + str(i + 2) + '.txt', delimiter=' ', header=None).values[0]
    output_reshape = np.reshape(output[0:-1], (1, 1296))
    commitment = np.where(output_reshape > 0.5, 1, -1)
    y = np.vstack([y, commitment])


    # print(lines.shape)
    # print(commitment.shape)
    # print(commitment[0])
    # print(type(commitment[0][0]))

"""
Create dataset of classification task with many redundant and few informative features
"""

X_train = X[0:800,:]
X_test = X[800:,:]
y_train = y[0:800,:]
y_test = y[800:,:]
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

result = np.zeros((199, 1296))
acc = np.zeros(1296)
diff_commit_count = 0

for commitment_number in range(1296):
    print('calculating commitment number:', commitment_number+1)
    y_train_col = y_train[:, commitment_number]

    if np.unique(y_train_col).shape[0] < 2:
        result[:, commitment_number] = y_train_col[0]
        acc[commitment_number] = 1
        continue
    else:
        diff_commit_count += 1
        count = 0
        svclassifier = SVC(kernel='linear')
        svclassifier.fit(X_train, y_train_col)

        y_test_col = y_test[:, commitment_number]
        y_pred = svclassifier.predict(X_test)

        # print(confusion_matrix(y_test_col, y_pred))
        # print(classification_report(y_test_col, y_pred))

        for i in range(len(y_pred)):
            if y_pred[i] == y_test_col[i]:
                count += 1
        acc[commitment_number] = count/len(y_pred)
        result[:, commitment_number] = y_pred

result = np.where(result > 0, 1, 0)

# result_output = np.reshape(result, (199, 24, 54))

print('different commitment counts: ', diff_commit_count)
np.savetxt(result_path + 'accuracy.txt', acc, delimiter=' ', fmt='%1.4e')
pred_count = 0
pred_sum = 0
for i in acc:
    if i != 1:
        pred_count += 1
        pred_sum += i
print('prediction accuracy:', pred_sum/pred_count)
print('average accuracy:', sum(acc)/1296)
for i in range(199):
    np.savetxt(result_path + str(i + 800) + '.txt', result[i], delimiter=' ', fmt='%d')

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

