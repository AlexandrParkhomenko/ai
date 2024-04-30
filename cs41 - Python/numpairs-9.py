"""
Created on Wed Apr 24 17:05:40 2024

@author: AlexandrParkhomenko

Откройте файл электронной таблицы, содержащей в каждой строке четыре натуральных числа. 
Определите количество строк таблицы, содержащих числа, для которых выполнены все условия:

    четыре числа строки можно разбить на две пары чисел с равными суммами
    максимальное число строки меньше суммы трёх оставшихся чисел
    сумма чисел в строке чётна

В ответе запишите только число.
"""

import itertools
#print(list(itertools.permutations([1, 2, 3, 4])))

result = 0

import numpy as np
t = np.genfromtxt("9.csv", delimiter=";", usemask=True)
for r in t:
    if (int(r[0]+r[1]+r[2]+r[3]) % 2) != 0:
        continue
    
    q=np.sort(r)
    if q[0]+q[1]+q[2] <= q[3]:
        continue
    
    a = True
    p = np.array(list(itertools.permutations(r)))
    #print(p)
    for s in p:
        #print(s)
        if s[0]+s[1] == s[2]+s[3]:
            #print(s)
            a = False
            break
    if a:
        continue
    result += 1
    #print(">>",r)
    
print(">>",result) #>> 139

#------------------------------------------------------------------------------

import csv
with open('9.csv') as f:
    reader = csv.reader(f, delimiter=';')
    data = [(int(col1), int(col2), int(col3), int(col4))
                for col1, col2, col3, col4 in reader]
result = 0
for r in data:
    if ((r[0]+r[1]+r[2]+r[3]) % 2) != 0:
        continue
    q=sorted(r)
    if q[0]+q[1]+q[2] <= q[3]:
        continue
    if (q[0]+q[1] == q[2]+q[3]) or \
       (q[0]+q[2] == q[1]+q[3]) or \
       (q[0]+q[3] == q[2]+q[1]):
        result += 1
print(result)
