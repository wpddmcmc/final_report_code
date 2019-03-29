import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tensorboardX import SummaryWriter

class Batch_Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):    # inherting torch Module
        super(Batch_Net, self).__init__()       # inherting __init__
        # hidden layer with activation function batch normalization
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        # output layer
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 3, 20)
            nn.Conv2d(
                in_channels=1,      # input height
                out_channels=16,    # n_filters
                kernel_size=(2,3),      # filter size
                stride=1,           # filter movement/step
                padding=1,          # padding=(kernel_size-1)/2 height and width don't change
            ),  # output shape (16, 2, 18)
            nn.ReLU(),              # activation
            nn.MaxPool2d(kernel_size=2),  # sample in 2x2 space, output shape (16, 1, 10)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 1, 10)
            nn.Conv2d(16, 32, 3, 1, 1),     # output shape (32, 1, 10)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(1),                # output shape (32, 1, 10)
        )
        self.out = nn.Linear(32 * 10, 11)  # fully connected layer, output 11 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # multi-dimension convolution graph (batch_size, 32 * 2 * 2)
        output = self.out(x)
        return output

class Net(torch.nn.Module):     # inherting torch Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # inherting __init__
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer linear output
        self.out = torch.nn.Linear(n_hidden, n_output)       # output layer linear output

    def forward(self, x):
        # forward propagation inputs data, neural network analysis prediction value
        x = F.relu(self.hidden(x))      # inspirit function(linear value of hidden layer)
        x = self.out(x)                 # output value
        return x

def readlabel():
    fr = open('lable.txt','r')     # read file
    data_y = np.zeros((20,10,2),dtype=int)                      # label data initialization
    squence = -1
    for line in fr:     # read line
        if line.find("_e",0,len(line))!=-1:
            squence=squence+1
        v = line.strip().split(' ')
        if(v[0]=='walk:'):
            data_y[squence][0][0] = int(v[1])
            data_y[squence][0][1] = int(v[2])
        elif (v[0]=='sitDown:'):
            data_y[squence][1][0] = int(v[1])
            data_y[squence][1][1] = int(v[2])
        elif (v[0]=='standUp:'):
            data_y[squence][2][0] = int(v[1])
            data_y[squence][2][1] = int(v[2])
        elif (v[0]=='pickUp:'):
            data_y[squence][3][0] = int(v[1])
            data_y[squence][3][1] = int(v[2])
        elif (v[0]=='carry:'):
            data_y[squence][4][0] = int(v[1])
            data_y[squence][4][1] = int(v[2])
        elif (v[0]=='throw:'):
            data_y[squence][5][0] = int(v[1])
            data_y[squence][5][1] = int(v[2])
        elif (v[0]=='push:'):
            data_y[squence][6][0] = int(v[1])
            data_y[squence][6][1] = int(v[2])
        elif (v[0]=='pull:'):
            data_y[squence][7][0] = int(v[1])
            data_y[squence][7][1] = int(v[2])
        elif (v[0]=='waveHands:'):
            data_y[squence][8][0] = int(v[1])
            data_y[squence][8][1] = int(v[2])
        elif (v[0]=='clapHands:'):
            data_y[squence][9][0] = int(v[1])
            data_y[squence][9][1] = int(v[2])
    return data_y

def readfiles(label):
    count=0
    path = ".\joints"  # path of files
    files = os.listdir(path)  # get file names
    s = []
    for file in files:
        if not os.path.isdir(file):  # only read files
            count = count+len(open(path + "//" + file).readlines())
    print(count)
    data_y = []  # label data
    data_x = torch.zeros(count, 20, 3)  # skeleton data
    linenum = 0
    fileindex=0
    for file in files:
        if not os.path.isdir(file):
            fr = open(path + "//" + file)
            for line in fr:
                v = line.strip().split('  ')
                if (int(v[0]) >= label[fileindex][0][0] and int(v[0]) < label[fileindex][0][1]):
                    data_y.append(1)
                elif (int(v[0]) >=  label[fileindex][1][0] and int(v[0]) <  label[fileindex][1][1]):
                    data_y.append(2)
                elif (int(v[0]) >=  label[fileindex][2][0] and int(v[0]) <  label[fileindex][2][1]):
                    data_y.append(3)
                elif (int(v[0]) >=  label[fileindex][3][0] and int(v[0]) <  label[fileindex][3][1]):
                    data_y.append(4)
                elif (int(v[0]) >=  label[fileindex][4][0] and int(v[0]) <  label[fileindex][4][1]):
                    data_y.append(5)
                elif (int(v[0]) >=  label[fileindex][5][0] and int(v[0]) <  label[fileindex][5][1]):
                    data_y.append(6)
                elif (int(v[0]) >=  label[fileindex][6][0] and int(v[0]) <  label[fileindex][6][1]):
                    data_y.append(7)
                elif (int(v[0]) >=  label[fileindex][7][0] and int(v[0]) <  label[fileindex][7][1]):
                    data_y.append(8)
                elif (int(v[0]) >=  label[fileindex][8][0] and int(v[0]) <  label[fileindex][8][1]):
                    data_y.append(9)
                elif (int(v[0]) >=  label[fileindex][9][0] and int(v[0]) <  label[fileindex][9][1]):
                    data_y.append(10)
                else:
                    data_y.append(0)
                for num in range(20):
                    data_x[linenum, num] = torch.tensor(
                        [float(v[3 * num + 1]), float(v[3 * num + 2]), float(v[3 * num + 3])])
                linenum = linenum + 1
            fileindex = fileindex+1
    data_y = torch.from_numpy(np.asarray(data_y, dtype=np.int64))
    print(data_y.size())
    print(data_x.size())
    return  data_x,data_y,count

label = readlabel()
print(np.shape(label))
data_x,data_y,count = readfiles(label)

print('Choose network:\n1.Linear Network\n2.Full Connection Network\n3.CNN')
key=input("Enter index to choose network:")
if key=='1':
    model = Net(n_feature=3*20, n_hidden=10, n_output=11)  # instantiate function neural network
if key == '2':
    model = Batch_Net(3 * 20, 300, 100, 11)
if key == '3':
    data_x = data_x.reshape(count, 1, 20, 3)
    model = CNN()    # instantiate convolution neural network

print(data_x.size())
print(data_y.size())
deal_dataset = TensorDataset(data_x, data_y)
train_loader = DataLoader(dataset=deal_dataset,batch_size=count)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)   # optimize all cnn parameters

for i, (inputs, labels) in enumerate(train_loader):
    if key=='1':
        inputs=inputs.reshape(-1,60)
        writer = SummaryWriter(comment='Linear_Net')
    if key == '2':
        inputs = inputs.reshape(-1, 60)
        writer = SummaryWriter(comment='Full_Connection_Net')
    if key == '3':
        writer = SummaryWriter(comment='CNN')
    writer.add_graph(model, inputs)

    for t in range(30001):
        out = model(inputs)  # feed net training data, get prediction
        loss = criterion(out, labels)  # calculate loss
        optimizer.zero_grad()  # clear last grad
        loss.backward()  # loss backward, calculate new data
        optimizer.step()  # add new weight to net parameters
        writer.add_scalar('Loss', loss, t)
        if t % 1000 == 0:
            prediction = torch.max(F.softmax(out), 1)[1]
            pred_y = prediction.data.numpy().squeeze()
            target_y = labels.data.numpy()
            accuracy = sum(pred_y == target_y) / count
            writer.add_scalar('Accuracy', accuracy, t)
            print('Step[{}], Loss:{} Accuracy:{}'.format(t, loss.item(), accuracy))
writer.close()