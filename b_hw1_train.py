import csv
import numpy as np
import math
data =[]
n_row = 0
for i in range(18):
    data.append([])
#read data
with open('b_hw1_data/train.csv','r',encoding='utf-8') as f:
    rows = csv.reader(f,delimiter=',')
    for row in rows:
        if (n_row != 0):
            for i in range(3, 27):
                if (row[i] != 'NR'):
                    data[(n_row - 1) % 18].append(float(row[i]))
                else:
                    data[(n_row - 1) % 18].append(float(0))
        n_row += 1
#parse data to (x,y)
x=[]
y=[]

#每个月份 m
for m in range(12):
    #每个月471组数据，每组18*9个元素
    for i in range(471):
        x.append([])
        #每笔data里有9个小时数据
        for h in range(9):
            #每个小时有18个维度
            for w in range(18):
                #组成163列，18*9
                x[m*471+i].append(data[w][480*m+i+h])
        y.append(data[9][480*m+9+i])

x=np.array(x)
# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

y=np.array(y)

x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

w = np.zeros(len(x[0]))
l_rate = 10
repeat = 10000
for i in range(repeat):
    hypo = np.dot(x,w)
    loss = hypo - y
    cost = np.sum(loss**2) / len(x)
    cost_a  = math.sqrt(cost)
    gra = np.dot(x_t,loss)
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - l_rate * gra/ada
    print ('iteration: %d | Cost: %f  ' % ( i,cost_a))

np.save('b_train_w',w)
