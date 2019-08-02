import os
final2 = os.getcwd() + '/Final2'

for filename in os.listdir(final2):    
    with open(final2 + '/' + filename, 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        f.writelines(line for line in lines if line.strip())
        f.truncate()