"""
Задача.

Решите проблему оптимизации, описанную ниже, с помощью любой Python-библиотеки на ваш выбор. 
Используйте два способа:

    аналитический якобиан и гессиан;
    конечно-разностную реализацию гессиана и якобиана.

Сравните результаты методов 1) и 2). Что вы можете сказать о них?

min x1x4(x1 + x2 + x3) + x3 — Задача

x1x2x3x4 ⩾ 25 — Ограничивающее неравенство

x1**2 + x2**2 + x3**2 + x4**2 = 40 — Ограничивающее равенство

1 ⩽ x1x2x3x4 ⩽ 5 — Границы

x0 = (1, 5, 5, 1)  — Инициализация

В пакете SciPy реализован отличный метод минимизации trust-constr, 
который умеет одновременно использовать якобиан, гессиан и ограничения области поиска. 
В машинном обучении аналитический вид функции неизвестен, следовательно, неизвестен и вид её производных. 
Поэтому вторую конфигурацию метода мы зададим «с завязанными глазами», 
выполняя только численное дифференцирование. Запустив прилагаемый код примера, 
вы получите схожие результаты в обоих случаях.
"""

#!/bin/python

# Usage: $ python ./exercise_9.py
#
# Sample output:
# Analytical Jacobian for x = ( 1.000000, 5.000000, 5.000000, 1.000000 ) is ( 12.000000, 1.000000, 2.000000, 11.000000 )
# Finite difference Jacobian for x = ( 1.000000, 5.000000, 5.000000, 1.000000 ) is ( 12.500000, 1.000000, 2.000000, 11.000000 )
#
# Analytical Hessian for x = ( 1.000000, 5.000000, 5.000000, 1.000000 ) is ( 2.000000, 1.000000, 1.000000, 12.000000 )
# Finite difference Hessian for x = ( 1.000000, 5.000000, 5.000000, 1.000000 ) is ( 2.000000, 1.000000, 1.000000, 12.500000 )
# 
# Analytical minimum value = 17.182585
# Analytical solution:
# x1 = 1.001332
# x2 = 4.969890
# x3 = 3.494571
# x4 = 1.444127
# Finite difference minimum value = 17.014018
# Finite difference solution:
# x1 = 1.000000
# x2 = 4.742999
# x3 = 3.821150
# x4 = 1.379408
#
# Observation: in this case, the "trust-constr" method was able to find a better solution with
# finite-difference Jacobian and Hessian.

import numpy as np
from scipy.optimize import minimize


def f(x) :
	return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]


def jacobian(x) :
	return np.array(( \
		2 * x[0] * x[3] + x[1] * x[3] + x[2] * x[3], \
		x[0] * x[3], \
		x[0] * x[3] + 1, \
		x[0] * x[0] + x[0] * x[1] + x[0] * x[2]  ))


'''
UserWarning: delta_grad == 0.0. Check if the approximated function is linear. 
If the function is linear better results can be obtained by defining the Hessian 
as zero instead of using quasi-Newton approximations.
  warn('delta_grad == 0.0. Check if the approximated '
'''
def hessian(x) :
	return np.array(( \
		(2 * x[3], x[3], x[3], 2 * x[0] + x[1] + x[2]), \
		(x[3], 0, 0, x[0]), \
		(x[3], 0, 0, x[0]), \
		(2 * x[0] + x[1] + x[2], x[0], x[0], 0) ))

def hessian_(x) :
	return np.array([[0,0,0,0]])


dx0 = 0.5
dx1 = 0.5
dx2 = 0.5
dx3 = 0.5


def jacobian_fdiff(x) :
	x0 = np.copy(x)
	x1 = np.copy(x)
	x2 = np.copy(x)
	x3 = np.copy(x)
	x0[0] = x0[0] + dx0
	x1[1] = x1[1] + dx1
	x2[2] = x2[2] + dx2
	x3[3] = x3[3] + dx3
	f_x = f(x)
	return np.array(( \
		(f(x0) - f_x) / dx0, \
		(f(x1) - f_x) / dx1, \
		(f(x2) - f_x) / dx2, \
		(f(x3) - f_x) / dx3 ))


def hessian_fdiff(x) :
	x0 = np.copy(x)
	x1 = np.copy(x)
	x2 = np.copy(x)
	x3 = np.copy(x)
	x0[0] = x0[0] + dx0
	x1[1] = x1[1] + dx1
	x2[2] = x2[2] + dx2
	x3[3] = x3[3] + dx3
	jacobian_fdiff_x = jacobian_fdiff(x)
	return np.array(( \
		(jacobian_fdiff(x0) - jacobian_fdiff_x) / dx0, \
		(jacobian_fdiff(x1) - jacobian_fdiff_x) / dx1, \
		(jacobian_fdiff(x2) - jacobian_fdiff_x) / dx2, \
		(jacobian_fdiff(x3) - jacobian_fdiff_x) / dx3 ))


def constraint1(x):
	return x[0] * x[1] * x[2] * x[3] - 25.0


def constraint2(x):
	return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 - 40


# initial guesses
n = 4
x0 = np.zeros(n)
x0[0] = 1.0
x0[1] = 5.0
x0[2] = 5.0
x0[3] = 1.0

# compare analytical and finite difference Jacobians for checking
jac = jacobian(x0)
print("Analytical Jacobian for x = ( %f, %f, %f, %f ) is ( %f, %f, %f, %f )" % \
	(x0[0], x0[1], x0[2], x0[3], jac[0], jac[1], jac[2], jac[3]))
jac = jacobian_fdiff(x0)
print("Finite difference Jacobian for x = ( %f, %f, %f, %f ) is ( %f, %f, %f, %f )" % \
	(x0[0], x0[1], x0[2], x0[3], jac[0], jac[1], jac[2], jac[3]))
print("")

# compare analytical and finite difference Hessians for checking (first row)
hess = hessian(x0)
print("Analytical Hessian for x = ( %f, %f, %f, %f ) is ( %f, %f, %f, %f )" % \
	(x0[0], x0[1], x0[2], x0[3], hess[0][0], hess[0][1], hess[0][2], hess[0][3]))
hess = hessian_fdiff(x0)
print("Finite difference Hessian for x = ( %f, %f, %f, %f ) is ( %f, %f, %f, %f )" % \
	(x0[0], x0[1], x0[2], x0[3], hess[0][0], hess[0][1], hess[0][2], hess[0][3]))
print("")

# optimize
b = (1.0, 5.0)
bnds = (b, b, b, b)
con1 = { 'type': 'ineq', 'fun': constraint1 } 
con2 = { 'type': 'eq', 'fun': constraint2 }
cons = ( [ con1, con2 ] )

# The only method available from scipy.optimize.minimize that can make use all of
# { boundaries, constraints, Jacobian and Hessian } in the same time is "trust-constr"
solution = minimize(f, x0, method = "trust-constr", bounds = bnds, constraints = cons, jac = jacobian, hess = hessian_)
x = solution.x

# show final objective
print("Analytical minimum value = %f" % f(x))

# print solution
print("Analytical solution:")
print("x1 = %f" % x[0])
print("x2 = %f" % x[1])
print("x3 = %f" % x[2])
print("x4 = %f" % x[3])

solution = minimize(f, x0, method = "trust-constr", bounds = bnds, constraints = cons, jac = jacobian_fdiff, hess = hessian_fdiff)
x = solution.x

# show final objective
print("Finite difference minimum value = %f" % f(x))

# print solution
print("Finite difference solution:")
print("x1 = %f" % x[0])
print("x2 = %f" % x[1])
print("x3 = %f" % x[2])
print("x4 = %f" % x[3])
