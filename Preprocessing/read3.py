import fileinput
import os

process = os.getcwd() + '/processed'
final = os.getcwd() + '/Final'

for filename in os.listdir(process):
    f = open(process +'/' + filename, 'r')
    g = open(final + '/' + filename, 'w')
    for line in f:
        if line.strip():
            g.write("\t".join(line.split()[:2]) + "\n")

    f.close()
    g.close()
