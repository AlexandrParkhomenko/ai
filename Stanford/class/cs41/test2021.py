#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 18:49:10 2021

@author: a

additional links:
  https://fpk.unn.ru/kursy-programmirovaniya/programmirovanie-na-python/

"""

'-'
# a=(2,3,4)
# print(sum(a,3)) #12

'-'
# # private metods with leading and trailing underscores

'+'
# print('88'.isnumeric()) #True

'+'
# alphabets = 'abcd'
# for i in range(len(alphabets)):
#     print(alphabets) # abcd a a a
#     alphabets = 'a'

'+'
# s="hello"
# print(type(s)) #<class 'str'>

'-'
# a=[1,4,3,5,2]
# b=[3,1,5,2,4]
# print(a==b) # False
# print(set(a)==set(b)) # True

'+'
# a={1:"A",2:"B",3:"C"}
# print(a.get(1,4)) #A

'-'
# d={"john":40,"peter":45}
# # d.delete("john") #wrong
# del d["john"]
# print(d)

'+'
# x=2
# for i in range(x):
#     x-=2
#     print(x) #0 -2

'+'
# print("xyyzxyzxzxyy".count('yy',2)) #1

'-'
# list1=[1,2,3]
# print(list1*2) #[1, 2, 3, 1, 2, 3]

'+'
# t=32.00
# [round((x-32)*5/9) for x in t] # TypeError: 'float' object is not iterable

'+'
# int('10.8') #ValueError: invalid literal for int() with base 10: '10.8'

'+'
# a=(1,2,3,4)
# del a
# print(a) # name 'a' is not defined

# 17 wrong
# z=set('abc')
# z.add('san')
# print(z) # {'c', 'b', 'san', 'a'}
# z.update(set(['p','q']))
# print(z) #{'c', 'p', 'q', 'b', 'san', 'a'}

'+'
# print('abcdef12'.replace('cd','12')) #ab12ef12

'+'
# print('abcdef'.find("cd")) #2

'-' # 20
# data=[13,56,17]
# data.append([87])
# data.append([45,67])
# print(data) # [13, 56, 17, [87], [45, 67]]
# # I mean
# import numpy as np
# data = np.array([13,56,17])
# data = np.append(data,[87])
# data = np.append(data,[45,67])
# print(data) # [13 56 17 87 45 67]

'-'
# def func(x):
#     def funk1():
#         print("Decorated")
#         x()
#     return funk1
# def funk2():
#     print("Ordinary")
# p=func(funk2)
# p() #Decorated Ordinary

'+'
# if(9<0) and (0<-9):
#     print('hello')
# elif (9>0) or False:
#     print("good")    #good
# else:
#     print("bad")

'+'
# B = [[1,1,1]
#      [2,2,2]
#      [3,3,3]
#      ] #TypeError
# C = [(1,1,1),
#      (2,2,2),
#      (3,3,3)
#      ]

'+'
# #print('There are %g %d birds.' %4 %'blue') #TypeError: not enough arguments for format string
# print('There are %d %s birds.' %(4,'blue')) #There are 4 blue birds.
# # print('There are %d %s birds.', 4,'blue') #There are %d %s birds. 4 blue
# # print('There are %d %s birds.' %[4,'blue']) #%d format: a number is required, not list

'+'
# print(list(enumerate([2,3]))) #[(0, 2), (1, 3)]

''
# a,b,c=1,2,3
# print(a,b,c) #1 2 3

'+'
# print("A", end = ' ')
# print("B", end = ' ')
# print("C", end = ' ')
# # A B C 

'+'
# a={1:"A",2:"B",3:"C"}
# del a
# print(a) #name 'a' is not defined

'+'
# def cube(x):
#     return x*x*x
# x=cube(3)
# print(x) #27

'+'
# A = [[1,2,3],
#       [4,5,6],
#       [7,8,9]]
# result = [A[i][i] for i in range(len(A))]
# print(result) #[1, 5, 9]

'+'
# list1=[1,2,3,4,5]
# list1.insert(3,5) # or add, append ?
# print(list1) # [1, 2, 3, 5, 4, 5]

'+'
# [print(x) for x in range(0,20) if (x%2==0)]
# #0 2 4 6 8 10 12 14 16 18

'-'
# print('yz90'.isalnum()) #True

'-'
# print(any([2>8, 4>2, 1>2])) #True #True if anything is right

'-'
# 1st = "never" # SyntaxError: invalid syntax

'-'
# print(all([2,4,0,6])) #False #True if not exist 0

'+'
# a={}
# a['a']=1
# a['b']=[2,3,4]
# print(a) # {'a': 1, 'b': [2, 3, 4]}

'---'
# x="abcdef"
# i="a"
# while i in x[:-1]:
#     print(i, end = " ") # infinity loop

'-'
# alphabets='abcd'
# for i in range(len(alphabets)):
#     alphabets[i].upper()
# print(alphabets) #abcd

'+'
# s=["silicon","valley","sf"]
# [print((w.upper(),len(w))) for w in s] #('SILICON', 7) ('VALLEY', 6) ('SF', 2)

'-'
# print('{:,}'.format(1112223334)) #1,112,223,334

'-'
# print("hello"*"World") #TypeError: can't multiply sequence by non-int of type 'str'
# print("hello"."World") #SyntaxError: invalid syntax
# print("hello".add("World")) #AttributeError: 'str' object has no attribute 'add'
# print("hello".__add__("World")) # helloWorld

'-'
# print(bool('False') and bool('False')) #True #bool('False')==True
# print(bool()) #False

'+'
# print('abcd'.translate('a'.maketrans('abc','bcd')))

'+' # поспешил
# for index in range(10):
#     if index==5:
#         break
#     else:
#         print(index)
# else:
#     print("Here")
# #0 1 2 3 4

'-'
# print("Hello {0!r} and {1!s}".format('john','doe')) #Hello 'john' and doe

'-'
# world1="Apple"
# world2="Apple"
# list1=[1,2,3]
# list2=[1,2,3]
# print(world1 is world2) #True
# print(list1 is list2)   #False

'-'
# print("abcdef".center(7,'1')) #1abcdef

'-'
# print(3*1**3) #3

'+'
# # any returns true if any key of dictionary is true

'+'
# data = {0:'a', 1:'b', 2:'c'}
# for i in data.keys():
#     print(data[i]) #a b c

'-'
# print("Hello {} and {}".format('john','doe')) #Hello john and doe

'+'
# a = set([1,2,3,4,5])
# b = set([3,4])
# print(b.issubset(a)) #True

'---'
# x=8
# print(x>>2) #2

'-'
# # data = ['ab', 'cd']
# # for i in data:
# #     print(data)
# #     data.append(i.upper()) # infinity loop
# # print(data)

'-'
# print('hello'.partition('lo')) #('hel', 'lo', '')

'-'
# t='%(a)s, %(b)s, %(c)s'
# print(t % dict(a='hello', b='world', c='universe')) #hello, world, universe

'+'
# print("Python is good".split()) #['Python', 'is', 'good']

'-'
# print(float(26//3+3/3)) #9.0

'-'
# a={3,4,5}
# a.update([1,2,3]) # Q: maybe error?
# print(a) #{1, 2, 3, 4, 5}

'+'
# print('new','line') #new line

'+'
# example = "snow world"
# example[3] = 's'
# print(example) #TypeError: 'str' object does not support item assignment

'---'
# print('abc XYZ'.capitalize()) #Abc xyz

'-'
# a=40
# b=30
# a=a^b
#  # print(a,b)
# b=a^b
#  # print(a,b)
# a=a^b
# print(a,b)

'-'
#print(2+4.00, 2**4.0) #6.0 16.0

'+'
# #print(float('13+34?)) #SyntaxError: EOL while scanning string literal

'+'
# print("Hello".replace('l','e')) #Heeeo

'-'
# data={0,1,2}
# for i in data:
#     print(data.add(i)) #None None None 

'---'
# python supports anonymous functions at runtime, using a constructions called
# LAMBDA!

'-'
# print('abcdefcdghcd'.split('cd',2)) #['ab', 'ef', 'ghcd']

'+'
# s1 = "hello world"
# s2 = "world"
# print(s1.__contains__(s2)) #True

'+'
# print(float(6+int(2.49)%2)) #6.0

'-'
# a=(-1,0,1)
# #[x for x<0 in a]
# #[x<0 in a]
# #[x in a for x<0]
# [print(x) for x in a if x<0] #-1

'+'
# print(7//5)

'-'
# t=(1,2,4,3)
# print(t[1:3]) #(2, 4)

'+'
# index = 1
# while True:
#     if index%2 == 0:
#         break
#     print(index)
#     index+=2     #infinity loop
#  #    if index > 10:
#  #        break

'-'
# x=3
# print(eval('x^2')) #1 # dont understand. This is XOR?
# print(eval('x&2'))
# print(eval('x|2'))

#     # x | y | x∧y | x∨y | x→y | x⊕y | x≡y |
#     # --------------------------------------
#     # 0 | 0 |  0  |  0  |  1  |  0  |  1  |
#     # 1 | 0 |  0  |  1  |  0  |  1  |  0  |
#     # 0 | 1 |  0  |  1  |  1  |  1  |  0  |
#     # 1 | 1 |  1  |  1  |  1  |  0  |  1  |
#     # --------------------------------------

'-'
# print('abc'.encode()) # b'abc'

'+'
# list1=[1,3]
# list2=list1
# list1[0]=4
# print(list2) #[4, 3]

'+'
# str="hello"
# print(str[:2])

'+'
# def print_this(message, times=1):
#     print(message*times)
# print_this('Hello') #Hello
# print_this('World', 5) #WorldWorldWorldWorldWorld

'---'
# s={4>3, 0, 3-3}
# print(all(s)) #False
# print(any(s)) #True

'-'
# print('abcdefcdghcd'.split('cd',0)) #['abcdefcdghcd']

'+'
# print(~101) #-102   # digital numbers property

'+'
# #Where is function defined?
# #Another function, Class, Module

'-'
# print(''.isdigit()) #False

'+'
# a={1:"A",2:"B",3:"C"}
# b=a.copy()
# b[2]="D"
# print(a) #{1: 'A', 2: 'B', 3: 'C'}

'-'
# lst=[3,4,6,1,2]
# lst[1:2]=[7,8]
# print(lst) #[3, 7, 8, 6, 1, 2]

'-'
# #first in list, last in sets  discard/dispose/pop/remove
# lst=[1, 2, 3]
# lst.pop()
# print(lst)
# sets = {1, 2, 3}
# sets.pop()
# print(sets)

'+'
# def maximum(x,y):
#     if x>y:
#         return x
#     elif x==y:
#         return "eq"
#     else:
#         return y
# print(maximum(2, 3)) #3

'+'
# d={"john":40,"peter":45}
# print("john" in d) #True

'+'
# a=[(2,4),(1,2),(3,9)]
# a.sort()
# print(a) #[(1, 2), (2, 4), (3, 9)]

'---'
# # use operator "in"
# lists =   [1, 2, 3]
# print(1 in lists) #True
# sets =  {1, 2, 3}
# print(1 in sets) #True
# dicts = {1:"A",2:"B",3:"C"}
# print(1 in dicts) #True

'-'
# def writer():
#     title = 'Sir'
#     name = (lambda x:title + ' ' + x) #TypeError: <lambda>() missing 1 required positional argument: 'x'
#     return name()

# who = writer()
# who('Artur')

'-'
# def decorator1(x):
#     def f1(a,b):
#         print("hello")
#         if b==0:
#             print("NO")
#             return
#         return decorator1(a, b)
#     return f1
# @decorator1
# def decorator1(a,b):
#     return a%b
# decorator1(4,0) #hello NO

'---'
# s={2,5,6,6,7}
# print(s) #{2,5,6,7}

'+'
# data = [[[1,2],[3,4]],[[5,6],[7,8]]]
# print(data[1][0][0]) #5

'---'
# print('The {} side {1} {2}'.format('bright','of','life')) #ValueError: cannot switch from automatic field numbering to manual field specification
# print('The {0} side {1} {2}'.format('bright','of','life')) #The bright side of life

'-'
# print('xyyzxyyzxyyxyy'.lstrip('xyy')) #zxyyzxyyxyy

'+'
# for i in [1,2,3,4][::-1]:
#     print(i) #4 3 2 1
 
'+'
# statement = "hello world"
# result=[(i.upper(),len(i)) for i in statement]
# print(result) #[('H', 1), ('E', 1), ('L', 1), ('L', 1), ('O', 1), (' ', 1), ('W', 1), ('O', 1), ('R', 1), ('L', 1), ('D', 1)]

'+'
# letters=list('HELLO')
# print('first={0[0]}, third={0[2]}'.format(letters)) #first=H, third=L

'+'
# t1=(1,2,4,3)
# t2=(1,2,3,4)
# print(t1<t2) #False

'+'
# print(min([3,5,25,1,3])) #1

'+'
# print([index.lower() for index in "HELLO"]) #['h', 'e', 'l', 'l', 'o']

'---'
# nums=set([1,1,2,3,3,3,4,4])
# print(len(nums)) #4

'-'
# a={'B':5,'A':9,'C':7}
# print(sorted(a)) #['A', 'B', 'C']

'-'
# x=456
# print("%-006d"%x) #456

'---'
# nums=[0,1,2,3]
# index=-2
# for index not in nums: #SyntaxError: invalid syntax
#     print(index)
#     index+=1

'+'
# lists =   [1, 2, 3]
# print(lists[-1]) #3

'+'
# import numpy as np
# m=np.arange(1,16)
# m=np.insert(m,2,4)
# m=m.reshape(4,4)
# print(m)
# for i in range(0,4):
#     print(m[i][1], end=" ") #2 5 9 13

'---'
# a=[[]]*3
# a[1].append(7)
# print(a) #[[7], [7], [7]]

'+++'
# x=-122
# print("-%006d"%x) #--00122

'---'
# a={4,5,6}
# b={2,8,6}
# print(a-b) #{4, 5}

'-'
# a={1,2,3}
# b=a.copy()
# b.add(4)
# print(a) #{1, 2, 3}

'-'
# names = ['Elton', 'Edith', 'Carl']
# if 'elton' in names:
#     print(1)
# else:
#     print(2)  # 2





