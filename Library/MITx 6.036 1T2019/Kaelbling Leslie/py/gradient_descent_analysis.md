we can think of our gradient descent step as simply a multiplication of the previous value by α on each step.


α          | cases of convergence, oscillation, and divergence
-----      | --------------------------------------------------------------------------------------------
**α>1**    | Gradient descent diverges without oscillation; z→∞
**α=1**    | z^k = z^0, so no gradient descent steps occur
**1>α≥0**  | α^∞ approaches 0, so gradient descent converges; z→0
**0>α>−1** | α^∞ approaches 0 while changing signs every step, so converges with some oscillation
**α=−1**   | At every step, the sign of z flips. Gradient descent oscillates between z^0 and -z^0 endlessly
**−1>α**   | Gradient descent diverges with oscillation, since z grows but the sign of z flips at every step

see https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week4/week4_lab/?child=first



