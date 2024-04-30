Complete the following sentences:
Data Conversion is a process of (a)	data from one format to another while data formatting is the organization of information according to preset specifications.
A CSV file is a delimited text file that uses, as a data format, a (b)	to separate values while a space delimited file uses a (c)	to separate them.

 A - transforming space comma
 B - converting comma space
 C - changing semicolon comma
 D - copying comma point

Refer to the information below: 

Original dictionary:
employee_ID = {'name':'Harry' , 'id':1009 , 'salary':[1200,1250,1250]}


Desired output:
{'name':'Harry' , 'id':1009 , 'salary':[1200, 1250, 1250] , 'sector':'AI'}


Complete the following code to give the desired output:


#missing statement here

print(employee_ID)

employee_ID = employee_ID.fromkeys('sector', 'AI')
employee_ID['sector'] = employee_ID.get('sector', 'AI')
employee_ID[sector] = employee_ID.keys(sector : AI)
employee_ID['sector'] = employee_ID.pop('AI')



x="John Watson"
y="Mycroft Holmes"
def one():
    global x
    x="Sherlock"

def two():
    y="Irene Adler"


print(x)
one()
print(x)
print(y)
two()
print(y)


Review the below code:


food = ['apple', 'orange', 'banana', 'pear']
list(enumerate(food))


Which of the following is the output of the above code?


[ (0, 1, 2, 3), ('apple', 'orange', 'banana', 'pear')]

[('apple', 0), ('orange', 1), ('banana', 2), ('pear', 3)]

[(0, 'apple'), (1, 'orange'), (2, 'banana'), (3, 'pear')]

[('apple', 'orange', 'banana', 'pear'), (0, 1, 2, 3)]



Review the below code:


import pandas as pd


dis_run = {"Day": "km", "Monday": 8, "Thursday": 6.2, "Saturday": 12.5}

mytable = pandas.Series(dis_run, index = ["Day","Monday", "Saturday"])


print(mytable)


What is the output of the above code?


Day              km
Monday         8
Saturday    12.5
Day              km
Monday         8
Saturday    12.5
dtype: object
Day              km
Monday         8
Thursday     6.2
Saturday    12.5
dtype: object

Traceback: (...)Error



A “Scipy” module is used for a/an  ______(a) processing in Python. In Scipy, a submodule used for image processing is ______(b).

(a) image, (b) scipy.image

(a) graphical, (b) scipy.img

(a) scientific, (b) scipy.ndimage

(a) digital image, (b) scipy.nimage



Select the valid iteration paradigms in for loop:

During the encounter of the continue statement, “for” loop stops iterating.

During the encounter of infinite loop, “for” loop stops iterating.

During the encounter of the break statement, “for” loop stops iterating.

During the encounter of try/except statements, “for” loop stops iterating.


Review the below code: 


import numpy as np

A = np.array([9, 15, 15, 3, 6], ndmin=3)
B = np.arange(5)
C = A - B
print(C.max())
print("Number of dimensions of array C is ", C.ndim)


What is the output of the above code?
12
Number of dimensions of array C is 2
14

Number of dimensions of array C is 3
10
Number of dimensions of array C is 3
15
Number of dimensions of array C is 2
