import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

current_path = os.getcwd()
input_path = current_path + '/data/Input/'

result_path = current_path + '/data/pred_results/'
line_path = current_path + '/data/Lines/'

scenario1 = np.zeros((1, 186))
scenario2 = np.zeros((1, 186))
scenario3 = np.zeros((1, 186))
scenario4 = np.zeros((1, 186))
scenario5 = np.zeros((1, 186))
for i in range(250):
    f = open(line_path + str(i + 1) + ".txt", "r")

    f.readline()
    line = f.readline().split()
    currentrow = np.zeros((1, 186))
    for num in line:
        currentrow[0, int(num)-1] = 1
    scenario1 = np.vstack([scenario1, currentrow])
    # print(scenario1)

    f.readline()
    f.readline()
    line = f.readline().split()
    currentrow = np.zeros((1, 186))
    for num in line:
        currentrow[0, int(num) - 1] = 1
    scenario2 = np.vstack([scenario2, currentrow])

    f.readline()
    f.readline()
    line = f.readline().split()
    currentrow = np.zeros((1, 186))
    for num in line:
        currentrow[0, int(num) - 1] = 1
    scenario3 = np.vstack([scenario3, currentrow])

    f.readline()
    f.readline()
    line = f.readline().split()
    currentrow = np.zeros((1, 186))
    for num in line:
        currentrow[0, int(num) - 1] = 1
    scenario4 = np.vstack([scenario4, currentrow])

    f.readline()
    f.readline()
    line = f.readline().split()
    currentrow = np.zeros((1, 186))
    for num in line:
        currentrow[0, int(num) - 1] = 1
    scenario5 = np.vstack([scenario5, currentrow])

    f.close()

scenario1 = np.delete(scenario1, 0, 0)
scenario2 = np.delete(scenario2, 0, 0)
scenario3 = np.delete(scenario3, 0, 0)
scenario4 = np.delete(scenario4, 0, 0)
scenario5 = np.delete(scenario5, 0, 0)


for i in range(1):
    lines = pd.read_csv(input_path + 'Input' + str(i + 1) + ".txt", delimiter='\t', header=None).values
    X = np.reshape(lines, (1, 4464))


for i in range(249):
    lines = pd.read_csv(input_path + 'Input' + str(i + 2) + ".txt", delimiter='\t', header=None).values
    X = np.vstack([X, np.reshape(lines, (1, 4464))])


    # print(lines.shape)
    # print(commitment.shape)
    # print(commitment[0])
    # print(type(commitment[0][0]))

"""
Create dataset of classification task with many redundant and few informative features

Scenario 1
"""
print('-' * 10 + 'Scenario1' + '-'*10)

X_train = X[0:200,:]
X_test = X[200:,:]
y_train = scenario1[0:200,:]
y_test = scenario1[200:,:]

result = np.zeros((50, 186))
acc = np.zeros(186)
diff_line_count = 0

for line_number in range(186):
    y_train_col = y_train[:, line_number]

    if np.unique(y_train_col).shape[0] < 2:
        result[:, line_number] = y_train_col[0]
        acc[line_number] = 1
        continue
    else:
        diff_line_count += 1
        count = 0
        svclassifier = SVC(kernel='linear')
        svclassifier.fit(X_train, y_train_col)

        y_test_col = y_test[:, line_number]
        y_pred = svclassifier.predict(X_test)

        # print(confusion_matrix(y_test_col, y_pred))
        # print(classification_report(y_test_col, y_pred))

        for i in range(len(y_pred)):
            if y_pred[i] == y_test_col[i]:
                count += 1
        acc[line_number] = count/len(y_pred)
        result[:, line_number] = y_pred

result = np.where(result > 0, 1, 0)

# result_output = np.reshape(result, (199, 24, 54))
print('different line counts: ', diff_line_count)
np.savetxt(result_path + 'accuracy1.txt', acc, delimiter=' ', fmt='%1.4e')
pred_count = 0
pred_sum = 0
for i in acc:
    if i != 1:
        pred_count += 1
        pred_sum += i
print('prediction accuracy:', pred_sum/pred_count)
print('average accuracy:', sum(acc)/186)

pred_lines1 = []

for i in range(50):
    pred_lines_index = np.where(result[i] == 1)
    pred_lines1.append(np.array(pred_lines_index)+1)

"""
Create dataset of classification task with many redundant and few informative features

Scenario 2
"""
print('-' * 10 + 'Scenario2' + '-'*10)
X_train = X[0:200,:]
X_test = X[200:,:]
y_train = scenario2[0:200,:]
y_test = scenario2[200:,:]

result = np.zeros((50, 186))
acc = np.zeros(186)
diff_line_count = 0

for line_number in range(186):
    y_train_col = y_train[:, line_number]

    if np.unique(y_train_col).shape[0] < 2:
        result[:, line_number] = y_train_col[0]
        acc[line_number] = 1
        continue
    else:
        diff_line_count += 1
        count = 0
        svclassifier = SVC(kernel='linear')
        svclassifier.fit(X_train, y_train_col)

        y_test_col = y_test[:, line_number]
        y_pred = svclassifier.predict(X_test)

        # print(confusion_matrix(y_test_col, y_pred))
        # print(classification_report(y_test_col, y_pred))

        for i in range(len(y_pred)):
            if y_pred[i] == y_test_col[i]:
                count += 1
        acc[line_number] = count/len(y_pred)
        result[:, line_number] = y_pred

result = np.where(result > 0, 1, 0)

# result_output = np.reshape(result, (199, 24, 54))
print('different line counts: ', diff_line_count)
np.savetxt(result_path + 'accuracy2.txt', acc, delimiter=' ', fmt='%1.4e')
pred_count = 0
pred_sum = 0
for i in acc:
    if i != 1:
        pred_count += 1
        pred_sum += i
print('prediction accuracy:', pred_sum/pred_count)
print('average accuracy:', sum(acc)/186)

pred_lines2 = []

for i in range(50):
    pred_lines_index = np.where(result[i] == 1)
    pred_lines2.append(np.array(pred_lines_index)+1)

"""
Create dataset of classification task with many redundant and few informative features

Scenario 3
"""
print('-' * 10 + 'Scenario3' + '-'*10)
X_train = X[0:200,:]
X_test = X[200:,:]
y_train = scenario3[0:200,:]
y_test = scenario3[200:,:]

result = np.zeros((50, 186))
acc = np.zeros(186)
diff_line_count = 0

for line_number in range(186):
    y_train_col = y_train[:, line_number]

    if np.unique(y_train_col).shape[0] < 2:
        result[:, line_number] = y_train_col[0]
        acc[line_number] = 1
        continue
    else:
        diff_line_count += 1
        count = 0
        svclassifier = SVC(kernel='linear')
        svclassifier.fit(X_train, y_train_col)

        y_test_col = y_test[:, line_number]
        y_pred = svclassifier.predict(X_test)

        # print(confusion_matrix(y_test_col, y_pred))
        # print(classification_report(y_test_col, y_pred))

        for i in range(len(y_pred)):
            if y_pred[i] == y_test_col[i]:
                count += 1
        acc[line_number] = count/len(y_pred)
        result[:, line_number] = y_pred

result = np.where(result > 0, 1, 0)

# result_output = np.reshape(result, (199, 24, 54))
print('different line counts: ', diff_line_count)
np.savetxt(result_path + 'accuracy3.txt', acc, delimiter=' ', fmt='%1.4e')
pred_count = 0
pred_sum = 0
for i in acc:
    if i != 1:
        pred_count += 1
        pred_sum += i
print('prediction accuracy:', pred_sum/pred_count)
print('average accuracy:', sum(acc)/186)

pred_lines3 = []

for i in range(50):
    pred_lines_index = np.where(result[i] == 1)
    pred_lines3.append(np.array(pred_lines_index)+1)


"""
Create dataset of classification task with many redundant and few informative features

Scenario 4
"""
print('-' * 10 + 'Scenario4' + '-'*10)
X_train = X[0:200,:]
X_test = X[200:,:]
y_train = scenario4[0:200,:]
y_test = scenario4[200:,:]

result = np.zeros((50, 186))
acc = np.zeros(186)
diff_line_count = 0

for line_number in range(186):
    y_train_col = y_train[:, line_number]

    if np.unique(y_train_col).shape[0] < 2:
        result[:, line_number] = y_train_col[0]
        acc[line_number] = 1
        continue
    else:
        diff_line_count += 1
        count = 0
        svclassifier = SVC(kernel='linear')
        svclassifier.fit(X_train, y_train_col)

        y_test_col = y_test[:, line_number]
        y_pred = svclassifier.predict(X_test)

        # print(confusion_matrix(y_test_col, y_pred))
        # print(classification_report(y_test_col, y_pred))

        for i in range(len(y_pred)):
            if y_pred[i] == y_test_col[i]:
                count += 1
        acc[line_number] = count/len(y_pred)
        result[:, line_number] = y_pred

result = np.where(result > 0, 1, 0)

# result_output = np.reshape(result, (199, 24, 54))
print('different line counts: ', diff_line_count)
np.savetxt(result_path + 'accuracy4.txt', acc, delimiter=' ', fmt='%1.4e')
pred_count = 0
pred_sum = 0
for i in acc:
    if i != 1:
        pred_count += 1
        pred_sum += i
print('prediction accuracy:', pred_sum/pred_count)
print('average accuracy:', sum(acc)/186)

pred_lines4 = []

for i in range(50):
    pred_lines_index = np.where(result[i] == 1)
    pred_lines4.append(np.array(pred_lines_index)+1)

"""
Create dataset of classification task with many redundant and few informative features

Scenario 5
"""
print('-' * 10 + 'Scenario5' + '-'*10)
X_train = X[0:200,:]
X_test = X[200:,:]
y_train = scenario5[0:200,:]
y_test = scenario5[200:,:]

result = np.zeros((50, 186))
acc = np.zeros(186)
diff_line_count = 0

for line_number in range(186):
    y_train_col = y_train[:, line_number]

    if np.unique(y_train_col).shape[0] < 2:
        result[:, line_number] = y_train_col[0]
        acc[line_number] = 1
        continue
    else:
        diff_line_count += 1
        count = 0
        svclassifier = SVC(kernel='linear')
        svclassifier.fit(X_train, y_train_col)

        y_test_col = y_test[:, line_number]
        y_pred = svclassifier.predict(X_test)

        # print(confusion_matrix(y_test_col, y_pred))
        # print(classification_report(y_test_col, y_pred))

        for i in range(len(y_pred)):
            if y_pred[i] == y_test_col[i]:
                count += 1
        acc[line_number] = count/len(y_pred)
        result[:, line_number] = y_pred

result = np.where(result > 0, 1, 0)

# result_output = np.reshape(result, (199, 24, 54))
print('different line counts: ', diff_line_count)
np.savetxt(result_path + 'accuracy5.txt', acc, delimiter=' ', fmt='%1.4e')
pred_count = 0
pred_sum = 0
for i in acc:
    if i != 1:
        pred_count += 1
        pred_sum += i
print('prediction accuracy:', pred_sum/pred_count)
print('average accuracy:', sum(acc)/186)

pred_lines5 = []

for i in range(50):
    pred_lines_index = np.where(result[i] == 1)
    pred_lines5.append(np.array(pred_lines_index)+1)


for i in range(50):
    file1 = open(result_path + str(i + 201) + ".txt", "w")
    file1.write("Scenario1 \n")
    answerlist = pred_lines1[i].tolist()[0]
    answer = ''
    for ans in answerlist:
        answer += str(ans) + ' '
    file1.write(answer)
    file1.write("\n")
    file1.write("\n")
    file1.write("Scenario2 \n")
    answerlist = pred_lines2[i].tolist()[0]
    answer = ''
    for ans in answerlist:
        answer += str(ans) + ' '
    file1.write(answer)
    file1.write("\n")
    file1.write("\n")
    file1.write("Scenario3 \n")
    answerlist = pred_lines3[i].tolist()[0]
    answer = ''
    for ans in answerlist:
        answer += str(ans) + ' '
    file1.write(answer)
    file1.write("\n")
    file1.write("\n")
    file1.write("Scenario4 \n")
    answerlist = pred_lines4[i].tolist()[0]
    answer = ''
    for ans in answerlist:
        answer += str(ans) + ' '
    file1.write(answer)
    file1.write("\n")
    file1.write("\n")
    file1.write("Scenario5 \n")
    answerlist = pred_lines5[i].tolist()[0]
    answer = ''
    for ans in answerlist:
        answer += str(ans) + ' '
    file1.write(answer)
    file1.write("\n")
    file1.write("\n")
    file1.close()

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

