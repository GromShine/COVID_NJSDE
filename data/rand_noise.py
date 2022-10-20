import numpy as np

def write_in_chunks(f, lst):
    for i in range(0, len(lst)):
        for j in range(len(lst[i])):
            lst[i][j]+=np.random.rand()
        chunk = lst[i]
        f.write(" ".join(str(val) for val in chunk) + "\n")

    
res = []
for line in open('m1_time.txt', errors='ignore'):
    a = line.split()
    res.append(a)

with open("output.txt", "w") as f:
    write_in_chunks(f, res)


