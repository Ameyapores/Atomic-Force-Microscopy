import os
from model2 import Network
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import statistics as st

path = os.getcwd()
df = pd.read_csv(path + '/data.csv')
files = df["Filename"].values.tolist()
Youngs= df['Modulus'].values.tolist()
final_list = list(zip(files,Youngs))
epoch_loss=[]
random.shuffle(final_list)
train = final_list[:5000]
Test = final_list[5000:]
net = Network()
net.cuda()

for epochs in range(10):

    random.shuffle(train)
    training = train[:2000]
    validation = train[4000:]

    a , b = zip(*training)
    c , d = zip(*validation)
    a, b , c , d = list(a), list(b), list(c), list(d)
    Loss_per_curve=[]
    
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    for filename in a:
        f=open(path + '/Final2' + '/' + filename,"r")
        lines=f.readlines()
        list1, list2 =[], []
        for x in lines:
            list1.append(float(x.split('\t')[0]))
            list2.append(float((x.split('\t')[1]).rstrip()))
        f.close()
        norm_x = [i/sum(list1)*100 for i in list1]
        norm_y = [i/sum(list2)*100 for i in list2]
        input_tensor= torch.tensor(norm_x + norm_y).type(torch.cuda.FloatTensor)
        output = net(input_tensor)
        target_out = torch.tensor(b[a.index(filename)]).type(torch.cuda.FloatTensor)
        loss = loss_fn(output, target_out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for filename in c:
        f=open(path + '/Final2' + '/' + filename,"r")
        lines=f.readlines()
        list1, list2 =[], []
        for x in lines:
            list1.append(float(x.split('\t')[0]))
            list2.append(float((x.split('\t')[1]).rstrip()))
        f.close()
        norm_x = [i/sum(list1)*100 for i in list1]
        norm_y = [i/sum(list2)*100 for i in list2]
        input_tensor= torch.tensor(norm_x + norm_y).type(torch.cuda.FloatTensor)
        #print (input_tensor.size(), filename)
        output = net(input_tensor)
        target_out = torch.tensor(d[c.index(filename)]).type(torch.cuda.FloatTensor)
        loss = loss_fn(output, target_out)
        #print(loss.cpu().numpy())
        Loss_per_curve.append(loss.cpu().detach().numpy())
    epoch_loss.append(np.mean(Loss_per_curve))

plt.plot(epoch_loss)
plt.show()
e, pankaj = zip(*Test)
e, pankaj = list(e), list(pankaj)
for filename in e:
    with open(path + '/Final2' + '/' + filename,"r") as f:
        lines=f.readlines()
        list1, list2 =[], []
        for x in lines:
            list1.append(float(x.split('\t')[0]))
            list2.append(float((x.split('\t')[1]).rstrip()))

        norm_x = [i/sum(list1)*100 for i in list1]
        norm_y = [i/sum(list2)*100 for i in list2]
    input_tensor= torch.tensor(norm_x + norm_y).type(torch.cuda.FloatTensor)
    target_out = pankaj[e.index(filename)]
    output = net(input_tensor)
    print(output, target_out)