
import csv
import os
import pandas as pd

path = os.getcwd()
df = pd.read_csv(path + '/new_data.csv')
files = df["Filename"].values.tolist()
x_values= df['x_value'].values.tolist()
Youngs = df['young'].values.tolist()
sum1 = sum(x_values)

with open(path + '/new_data_proc.csv', 'a', newline='') as sfile:
    for i in range(len(Youngs)):
        data = [files[i], x_values[i]/sum1, Youngs[i]] 
        writer = csv.writer(sfile)
        writer.writerows([data])
    
            