# 6.1) analytic gradient descent )

import numpy as np

"""## 6) Implementing gradient descent
In this section we will implement generic versions of gradient descent and apply these to the SVM objective.

<b>Note: </b> If you need a refresher on gradient descent,
you may want to reference
<a href="https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week4/gradient_descent/2">this week's notes</a>.

### 6.1) Implementing Gradient Descent
We want to find the $x$ that minimizes the value of the *objective
function* $f(x)$, for an arbitrary scalar function $f$.  The function
$f$ will be implemented as a Python function of one argument, that
will be a numpy column vector.  For efficiency, we will work with
Python functions that return not just the value of $f$ at $f(x)$ but
also return the gradient vector at $x$, that is, $\nabla_x f(x)$.

We will now implement a generic gradient descent function, `gd`, that
has the following input arguments:

* `f`: a function whose input is an `x`, a column vector, and
  returns a scalar.
* `df`: a function whose input is an `x`, a column vector, and
  returns a column vector representing the gradient of `f` at `x`.
* `x0`: an initial value of $x$, `x0`, which is a column vector.
* `step_size_fn`: a function that is given the iteration index (an
  integer) and returns a step size.
* `max_iter`: the number of iterations to perform

Our function `gd` returns a tuple:

* `x`: the value at the final step
* `fs`: the list of values of `f` found during all the iterations (including `f(x0)`)
* `xs`: the list of values of `x` found during all the iterations (including `x0`)

**Hint:** This is a short function!

**Hint 2:** If you do `temp_x = x` where `x` is a vector
(numpy array), then `temp_x` is just another name for the same vector
as `x` and changing an entry in one will change an entry in the other.
You should either use `x.copy()` or remember to change entries back after modification.

Some utilities you may find useful are included below.
"""

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

"""The main function to implement is `gd`, defined below."""

def gd(f, df, x0, step_size_fn, max_iter):
    x  = np.copy(x0)
    xs = []
    fs = []
    t = 0;
    while t < max_iter:
        t += 1
        y  = f(x)
        dy = df(x)
        xs.append(x)
        fs.append(y)
        x = x - step_size_fn(t) * dy
        #if np.dot(dy.T,dy) == 0:
        #    break
    return (x,fs,xs)

# draft
def gd_draft(f, df, x0, step_size_fn, max_iter):    
    eta = step_size_fn
    theta = x0
    epsilon = 0.1
    theta0 = np.copy(theta)
    t = 0
    while np.abs(f(theta) - f(theta0)) < epsilon:
        theta0 = np.copy(theta)
        t = t + 1
        s =  eta(t)
        print(s)
        theta = theta0 - s * df(theta)
        if t >= max_iter:
            break
#    return (x,fs,xs)

"""To evaluate results, we also use a simple `package_ans` function,
which checks the final `x`, as well as the first and last values in
`fs`, `xs`.
"""

def package_ans(gd_vals):
    x, fs, xs = gd_vals
    return [x.tolist(), 
            [fs[0], fs[-1]], 
            [xs[0].tolist(), 
             xs[-1].tolist()]]

"""The test cases are provided below, but you should feel free (and are encouraged!) to write more of your own."""

# Test case 1
#ans=package_ans(gd(f1, df1, cv([0.]), lambda i: 0.1, 1000))

# Test case 2
ans=package_ans(gd(f2, df2, cv([0., 0.]), lambda i: 0.01, 1000))
print(ans)
