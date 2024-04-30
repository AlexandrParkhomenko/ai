# 6.2) Numerical Gradient

import numpy as np

def rv(value_list):
    return np.array([value_list])

def cv(value_list):
    return np.transpose(rv(value_list))

def f1(x):
    return float((2 * x + 3)**2)

def df1(x):
    return 2 * 2 * (2 * x + 3)

def f2(v):
    x = float(v[0]); y = float(v[1])
    return (x - 2.) * (x - 3.) * (x + 3.) * (x + 1.) + (x + y -1)**2

def df2(v):
    x = float(v[0]); y = float(v[1])
    return cv([(-3. + x) * (-2. + x) * (1. + x) + \
               (-3. + x) * (-2. + x) * (3. + x) + \
               (-3. + x) * (1. + x) * (3. + x) + \
               (-2. + x) * (1. + x) * (3. + x) + \
               2 * (-1. + x + y),
               2 * (-1. + x + y)])


def num_grad(f, delta=0.001):
    def df(x):
        X = np.copy(x)
        fs = np.array([])
        for i in range(X.shape[0]):
            d = np.zeros(X.shape)
            d[i] = delta
            fs = np.append(fs, ((
                f(X+d)
                -f(X-d)
                )/(2*delta)
                ))
        return cv(fs.T)
    return df

"""The test cases are shown below; these use the functions defined in the previous exercise."""

x = cv([0.])
ans=(num_grad(f1)(x).tolist(), x.tolist())
print("ans:", ans)

x = cv([0.1])
ans=(num_grad(f1)(x).tolist(), x.tolist())
print("ans:", ans)

x = cv([0., 0.])
ans=(num_grad(f2)(x).tolist(), x.tolist())
print("ans:", ans)

x = cv([0.1, -0.1])
ans=(num_grad(f2)(x).tolist(), x.tolist())
print("ans:", ans)
