# Game Theory II: Notes

```python
# Borda Count
import numpy as np
a = np.array([["B", "C", "A", "D"],
              ["B", "D", "C", "A"],
              ["D", "C", "A", "B"],
              ["A", "D", "B", "C"],
              ["A", "D", "C", "B"]])
j_max,i_max = np.shape(a)

d={}
for j in range(0,j_max):
    for i in range(0,i_max):
        e = a[j][i]
        if e not in d:
            d[e]=0
        d[e]=d[e]+i_max-1-i
print("winner is",max(d, key=d.get),":",d)
```
winner is D : {'B': 7, 'C': 6, 'A': 8, 'D': 9}

