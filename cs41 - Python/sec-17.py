#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 15:14:37 2024

@author: AlexandrParkhomenko
"""

# Доступен файл для чтения: 17.txt
f = open('17.txt', 'r')
c = f.read().split('\n')
#print(c)
m = int(c[0])
#print(m)
for x in c:
  if x=='': continue
  x = int(x)
  if(x % 21) != 0: continue
  if x > m:
    m=x

print(">",m)

o = 0
co = 0
su = 100000*2
for x in c:
  if x=='': continue
  x = int(x)
  if(o>m or x>m):
      print(o,x)
      co += 1
      t = o+x
      if t < su: su = t
  o = x
print(co,su)
