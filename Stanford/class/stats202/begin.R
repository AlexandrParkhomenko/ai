# Suppose that the number of web hits to a particular site are approximately normally distributed with a mean of 100 hits per day and a standard deviation of 10 hits per day. What's the probability that a given day has fewer than 93 hits per day expressed as a percentage to the nearest percentage point?
round(pnorm(93, mean = 100, sd = 10) * 100)

# Consider the following pmf given in R
p <- c(.1, .2, .3, .4)
x <- 2 : 5

# What is the variance?
sum(x^2 * p) - sum(x * p)^2
