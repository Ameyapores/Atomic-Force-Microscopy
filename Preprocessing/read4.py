import fileinput
import os

final = os.getcwd() + '/Final'
final2 = os.getcwd() + '/Final2'

for filename in os.listdir(final):
    g = open(final2 + '/' + filename, 'w')
    with open(final + '/' + filename) as f:
        data = f.readlines()
    g.write("\n".join(data[-250:]))
    g.close()
    

    