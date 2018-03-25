import csv
import numpy as np
w = np.load('b_train_w.npy')
#读取test

n_row = 0
test_x = []
with open('b_hw1_data/test.csv','r',encoding='utf-8') as f:
    rows = csv.reader(f,delimiter=',')
    for row in rows:
        if(n_row%18==0):
            test_x.append([])
            for i in range(2,11):
                test_x[n_row // 18].append(float(row[i]))
        else:
            for i in range(2,11):
                if(row[i]!='NR'):
                    test_x[n_row//18].append(float(row[i]))
                else:
                    test_x[n_row//18].append(float(0))
        n_row +=1
test_x = np.array(test_x)
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)
print(test_x,test_x.shape)

ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(test_x[i],w)
    ans[i].append(a)

filename = "b_predict.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()