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
from scipy.optimize import curve_fit

path = os.getcwd()
df = pd.read_csv(path + '/new_data_proc.csv')
files = df["Filename"].values.tolist()
x_value = df['x_value'].values.tolist()
Youngs = df['young'].values.tolist()
final_list = list(zip(files, x_value))
epoch_loss=[]
random.shuffle(final_list)
train = final_list[:5000]
Test = final_list[5000:]
net = Network()
net.cuda()

def func(y, a, c):
    return  a * pow(y, 1.5) + c

for epochs in range(25):

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
        norm_x = [i/sum(list1) for i in list1]
        norm_y = [i/sum(list2) for i in list2]
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
        norm_x = [i/sum(list1) for i in list1]
        norm_y = [i/sum(list2) for i in list2]
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

        norm_x = [i/sum(list1) for i in list1]
        norm_y = [i/sum(list2) for i in list2]
    input_tensor= torch.tensor(norm_x + norm_y).type(torch.cuda.FloatTensor)
    target_out = pankaj[e.index(filename)] * (-0.0016982442773780034)
    output = net(input_tensor)
    x_value_output= output.cpu().detach().numpy() * (-0.0016982442773780034)
    print (x_value_output, target_out)
    
    for i in range(len(norm_x)-1):
        if x_value_output <= list1[::-1][i]:
            x_values = list1[i:]
            y_values = list2[i:]
            print("hi")
            break
        else:
            print("No such x")
    new_x_value = [abs(x) for x in x_values]
    (a, c), pocv = curve_fit(func, new_x_value, y_values)
    pred_young = a/5.62e-3
    act_young = Youngs[files.index(filename)]
    print(pred_young, act_young)