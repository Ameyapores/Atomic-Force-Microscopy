import os
import shutil

directory = os.getcwd() + '/all'
process = os.getcwd() + '/processed'

for filename in os.listdir(directory):
    with open(directory + '/'+ filename) as read:
        for n in range(76):
            read.readline()
        with open(process+'/'+ filename, 'w') as write:
                shutil.copyfileobj(read, write)
    