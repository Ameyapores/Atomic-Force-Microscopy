
import fileinput
import os

process = os.getcwd() + '/processed'

for filename in os.listdir(process):
    for linenum,line in enumerate(open(process +'/' + filename)):
        if "index:" in line:
            N = linenum
            break
        else:
            N=0
    #print (N)
    fin = fileinput.FileInput(process +'/' + filename, inplace=1)
    counter = 0
    for line in fin:
        print (line)
        counter += 1
        if counter == N: 
                break

        