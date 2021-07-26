# Suppose that the number of web hits to a particular site are approximately normally distributed with a mean of 100 hits per day and a standard deviation of 10 hits per day. What's the probability that a given day has fewer than 93 hits per day expressed as a percentage to the nearest percentage point?
round(pnorm(93, mean = 100, sd = 10) * 100)

# Consider the following pmf given in R
p <- c(.1, .2, .3, .4)
x <- 2 : 5

# What is the variance?
sum(x^2 * p) - sum(x * p)^2

# Load the data set mtcars in the datasets R package. Calculate a 95% confidence interval to the nearest MPG for the variable mpg.
round(t.test(sort(mtcars$mpg))$conf)

# (basic)[https://github.com/DataScienceSpecialization/courses]

# Consider the mtcars dataset. Construct a 95% T interval for MPG comparing 4 to 6 cylinder cars (subtracting in the order of 4 - 6) assume a constant variance.
m4 <- mtcars$mpg[mtcars$cyl == 4]
m6 <- mtcars$mpg[mtcars$cyl == 6]
mtcars46 = mtcars[mtcars$cyl %in% c(4,6), ]
mtcars46[order(mtcars46$cyl),,drop=FALSE]
#this does 4 - 6
confint <- as.vector(t.test(m4, m6, var.equal = TRUE)$conf.int)

# Bayes' Formula
# P(B|A) = P(B&A)/P(A) = P(A|B)*P(B)/P(A) = P(A|B)*P(B)/( P(A|B)*P(B) + P(A|~B)*P(~B) )


# http://www.sci.utah.edu/~arpaiva/classes/UT_ece3530/hypothesis_testing.pdf

