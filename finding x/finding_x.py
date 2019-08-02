import os
import pandas as pd
from scipy.optimize import curve_fit, least_squares
import numpy as np
import math 
import matplotlib.pyplot as plt
import csv

def func(y, a, c):
    return  a * pow(y, 1.5) + c

path = os.getcwd()

'''title = ['Filename','x_value', 'young']
with open(path + '/new_data.csv', 'a', newline='') as sfile:
    writer = csv.writer(sfile)
    writer.writerow(title)'''   

df = pd.read_csv(path + '/data.csv')
files = df["Filename"].values.tolist()
Youngs= df['Modulus'].values.tolist()

for filename in files:
    f=open(path + '/Final2' + '/' + filename,"r")
    lines=f.readlines()
    list1, list2 =[], []

    for x in lines:
        list1.append(float(x.split('\t')[0]))
        list2.append(float((x.split('\t')[1]).rstrip()))
    f.close()
    newL1, newL2 = list1[:], list2[:]
    for item in list1:
        if item >= 0:
            newL1.remove(item)
            newL2.remove(list2[list1.index(item)])
    
    newL3 = [abs(x) for x in newL1]
    young = Youngs[files.index(filename)]

    for i in range(len(newL1)-20):
        sand, aasd = newL3[::-1][i:], newL2[::-1][i:]
        (a, c), pocv = curve_fit(func, sand, aasd)
        
        pred_young = a/5.62e-3
        #print(pred_young, young, a, c)
        if abs(pred_young-young)<=10.:
            print("The value of x is", newL1[::-1][i], pred_young, young)
            y_fit = func(np.array(newL3), a, c)
            data = [filename, newL1[::-1][i], young]
            '''with open(path + '/new_data.csv', 'a', newline='') as sfile:
                writer = csv.writer(sfile)
                writer.writerows([data])'''
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.plot(newL1, newL2, '--k')
            ax.plot(newL1, y_fit, '-')
            plt.show()
            break
        elif pred_young < young and pred_young> young-150:
            y_fit = func(np.array(newL3), a, c)         
            print("The value of x is", newL1[::-1][i], pred_young, young)   
            data = [filename, newL1[::-1][i], young]
            '''with open(path + '/new_data.csv', 'a', newline='') as sfile:
                writer = csv.writer(sfile)
                writer.writerows([data])'''
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.plot(newL1, newL2, '--k')
            ax.plot(newL1, y_fit, '-')
            plt.show()
            break
