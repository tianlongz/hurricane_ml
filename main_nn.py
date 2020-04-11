import os

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable

import pandas as pd
import numpy as np

current_path = os.getcwd()
input_path = current_path + '/data/Input/'
output_path = current_path + '/data/ML/Results/'

# input_data = 
# label_data = 

for i in range(1):
    lines = pd.read_csv(input_path + 'Input' + str(i + 1) + ".txt", delimiter='\t', header=None).values
    lines = np.reshape(lines, (1,24,186))
    X = Variable(torch.tensor(lines).type('torch.DoubleTensor'))

    output = pd.read_csv(output_path + 'Commitment' + str(i + 1) + '.txt', delimiter=' ', header=None).values[0]
    output_reshape = np.reshape(output[0:-1], (1,24, 54))

    # commitment = np.where(output_reshape > 0.5, 1, 0)
    y = Variable(torch.tensor(output_reshape).type('torch.DoubleTensor'))

for i in range(1,2):
    lines = pd.read_csv(input_path + 'Input' + str(i + 1) + ".txt", delimiter='\t', header=None).values
    lines = np.reshape(lines, (1,24,186))
    X_test = Variable(torch.tensor(lines).type('torch.DoubleTensor'))

    output = pd.read_csv(output_path + 'Commitment' + str(i + 1) + '.txt', delimiter=' ', header=None).values[0]
    output_reshape = np.reshape(output[0:-1], (1,24, 54))

    # commitment = np.where(output_reshape > 0.5, 1, 0)
    y_test = Variable(torch.tensor(output_reshape).type('torch.DoubleTensor'))

# for i in range(100):
#     lines = pd.read_csv(input_path + 'Input' + str(i + 2) + ".txt", delimiter='\t', header=None).values
#     X_test= np.vstack([X, lines])

#     output = pd.read_csv(output_path + 'Commitment' + str(i + 2) + '.txt', delimiter=' ', header=None).values[0]
#     output_reshape = np.reshape(output[0:-1], (24, 54))

#     # commitment = np.where(output_reshape > 0.5, 1, 0)
#     y_test = np.vstack([y, output_reshape])



# Hyper Parameters
EPOCH = 1  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 1
LR = 0.001  # learning rate

'''
# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# pick 2000 samples to speed up testing
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[
         :2000] / 255.  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]
'''

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=186,
            hidden_size=54,
            num_layers=1,
        )
        self.out = nn.Linear(54, 2)  # fully connected layer, output 10 classes

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        # r_out = [BATCH_SIZE, input_size, hidden_size]
        # r_out[:, -1, :] = [BATCH_SIZE, hidden_size]  '-1'，表示选取最后一个时间点的 r_out 输出
        out = self.out(r_out[:, -1, :])
        # out = [BATCH_SIZE, 10]
        return out

rnn = RNN().float()

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

for epoch in range(EPOCH):
    # for step, (x, b_y) in enumerate(train_loader):  # gives batch data
        # b_x = x.view(-1, 24, 28)  # reshape x to (batch, time_step, input_size)

    output = rnn(X)  # rnn output
    loss = loss_func(output, y)  # cross entropy loss
    optimizer.zero_grad()  # clear gradients for this training step
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()

# test_output = rnn(test_x[:10].view(-1, 28, 28))
# pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
test_output = rnn(X_test)

# print(pred_y, 'prediction number')
# print(test_y[:10], 'real number')

print(test_output, 'prediction number')
print(y_test, 'real number')
