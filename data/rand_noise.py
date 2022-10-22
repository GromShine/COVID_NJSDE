import numpy as np
    
lst = []

for line in open('m1_time.txt', errors='ignore'):
    a = line.split()
    lst.append(a)


dict = {}
ans = []
with open("n1_time.txt", "w") as f:
    
    for i in range(0, len(lst)):
        for j in range(len(lst[i])):
            lst[i][j] = float(lst[i][j])
            lst[i][j] += 0.1*np.random.rand()
            lst[i][j] = round(lst[i][j],3)
            dict[lst[i][j]] = i+1
            ans.append(lst[i][j])
    ans.sort()
    print(ans)
    f.write(" ".join(str(val) for val in ans) + "\n")
    #chunk = lst[i]

ans_mark=[]
with open("n1_event.txt", "w") as f:
    for i in range(len(ans)):
        ans_mark.append(dict[ans[i]])
    print(ans_mark)
    f.write(" ".join(str(val) for val in ans_mark) + "\n")
    #chunk = lst[i]
    


