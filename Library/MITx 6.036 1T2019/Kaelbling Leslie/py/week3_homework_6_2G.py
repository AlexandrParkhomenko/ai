# 6.2G) (Optional)  Feel free to classify other images from each other.
# Which combinations perform the best, and which perform the worst?
# Do these make sense? Other than row and column average, are there any other features
# you could think of that would preserve some spatial information?

#-------------------------------------------------------------------------------
# Analyze MNIST data
#-------------------------------------------------------------------------------

# Your code here to process the MNIST data

result = {}
for i in range(10):
    for j in range(i+1,10):
        print(i,j)
        # HINT: change the [0] and [1] if you want to access different images
        d0 = mnist_data_all[i]["images"]
        d1 = mnist_data_all[j]["images"]
        y0 = np.repeat(-1, len(d0)).reshape(1,-1)
        y1 = np.repeat(1, len(d1)).reshape(1,-1)
        
        # data goes into the feature computation functions
        data = np.vstack((d0, d1))
        # labels can directly go into the perceptron algorithm
        labels = np.vstack((y0.T, y1.T)).T
        acc = hw3.get_classification_accuracy(raw_mnist_features(data), labels)
        result.update({i*10+j:acc})
print(result)
# best
# 14  0.98125
# 57  0.98125
# 01  0.975

# worst
# 35  0.57583
# 18  0.525
# 46  0.50792
# 49  0.48417
# 58  0.4825
