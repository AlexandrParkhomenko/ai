# ACHTUNG! NOT CORRECT ANSWER!
# This is draft

# What are the 10 most positive words in the dictionary, that is, the words that contribute most to a positive prediction?

import pdb
import numpy as np
import code_for_hw3_part2 as hw3

def perceptron(data, labels, params = {}, hook = None):
    # if T not in params, default to 100
    T = params.get('T', 50)
    (d, n) = data.shape

    theta = np.zeros((d, 1)); theta_0 = np.zeros((1, 1))
    for t in range(T):
        for i in range(n):
            x = data[:,i:i+1]
            y = labels[:,i:i+1]
            if y * positive(x, theta, theta_0) <= 0.0:
                theta = theta + y * x
                theta_0 = theta_0 + y
                if hook: hook((theta, theta_0))
    return theta, theta_0

def averaged_perceptron(data, labels, params = {}, hook = None):
    T = params.get('T', 50)
    (d, n) = data.shape

    theta = np.zeros((d, 1)); theta_0 = np.zeros((1, 1))
    theta_sum = theta.copy()
    theta_0_sum = theta_0.copy()
    for t in range(T):
        for i in range(n):
            x = data[:,i:i+1]
            y = labels[:,i:i+1]
            if y * positive(x, theta, theta_0) <= 0.0:
                theta = theta + y * x
                theta_0 = theta_0 + y
                if hook: hook((theta, theta_0))
            theta_sum = theta_sum + theta
            theta_0_sum = theta_0_sum + theta_0
    theta_avg = theta_sum / (T*n)
    theta_0_avg = theta_0_sum / (T*n)
    if hook: hook((theta_avg, theta_0_avg))
    return theta_avg, theta_0_avg

  
def eval_classifier(learner, data_train, labels_train, data_test, labels_test, params):
    th, th0 = learner(data_train, labels_train, params)
    return score(data_test, labels_test, th, th0)/data_test.shape[1]

def positive(x, th, th0):
    return np.sign(th.T@x + th0)

def score(data, labels, th, th0):
    return np.sum(positive(data, th, th0) == labels)

def xval_learning_alg(learner, data, labels, params, k):
    _, n = data.shape
    idx = list(range(n))
    np.random.seed(0)
    np.random.shuffle(idx)
    data, labels = data[:,idx], labels[:,idx]

    score_sum = 0
    s_data = np.array_split(data, k, axis=1)
    s_labels = np.array_split(labels, k, axis=1)
    for i in range(k):
        data_train = np.concatenate(s_data[:i] + s_data[i+1:], axis=1)
        labels_train = np.concatenate(s_labels[:i] + s_labels[i+1:], axis=1)
        data_test = np.array(s_data[i])
        labels_test = np.array(s_labels[i])
        score_sum += eval_classifier(learner, data_train, labels_train,
                                              data_test, labels_test, 
                                              params
                                              )
    return score_sum/k

#perceptron(data, labels, params = {'T':100}, hook = None)
#-------------------------------------------------------------------------------
# Review Data
#-------------------------------------------------------------------------------

# Returns lists of dictionaries.  Keys are the column names, 'sentiment' and 'text'.
# The train data has 10,000 examples
review_data = hw3.load_review_data('reviews.tsv')

# Lists texts of reviews and list of labels (1 or -1)
review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))

# The dictionary of all the words for "bag of words"
dictionary = hw3.bag_of_words(review_texts)

# The standard data arrays for the bag of words
review_bow_data = hw3.extract_bow_feature_vectors(review_texts, dictionary)
review_labels = hw3.rv(review_label_list)
print('review_bow_data and labels shape', review_bow_data.shape, review_labels.shape)

#-------------------------------------------------------------------------------
# Analyze review data
#-------------------------------------------------------------------------------

# Your code here to process the review data
theta_avg, theta_0_avg = averaged_perceptron(review_bow_data, review_labels, params = {'T':10}, hook = None)

stop_words = np.loadtxt("./stopwords.txt", dtype=str, skiprows=0)
#for w in stop_words:
#    del(dictionary[w])

def one_hot(x, k):
    r=np.zeros(k)
    r[x-1]=1
    return np.array([r])

#dictionary = np.array([dictionary])
entries = len(dictionary)
#zz = 0
result = {} #np.array([])
for word in dictionary:
    # np.sign
    x =  one_hot(list(dictionary.keys()).index(word), entries)
    #print(theta_avg.shape, x.shape)
    if word in stop_words:
        continue
    else:
        result.update({float(x@theta_avg + theta_0_avg):word})
   # result = np.append(result,)
    #zz +=1
    #if zz > 10:
    #    break
#result = result.reshape((result.shape[0])//2,2)
print(result)
# ('do','watching','low','smooth','juices','individual','boy','wild','versatility','flu','nature')
# ( 'do', 'watching', 'low', 'smooth', 'range', 'saying', 'room', 'workers', 'job', 'given' )
# ( 'feel', 'flavor', 'amazon', 'coffee', 'range', 'down', 'stopping', 'some', 'them', 'contain' )
