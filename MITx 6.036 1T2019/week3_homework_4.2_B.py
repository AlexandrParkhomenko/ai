# (Optional) Is there any set of two features you can use to attain comparable results
# as your best accuracy? What are they?

# import code_for_hw3_part2.py
# T = 1

auto_data = load_auto_data("./auto-mpg.tsv")
features = [('cylinders',one_hot),('displacement',standard),('horsepower',standard),('weight',standard),('acceleration',standard),('origin',one_hot)]
result = np.array([])
for i in range(len(features)):
    for j in range(i+1,len(features)):
        f = [(features[i]),(features[j])]
        data, labels = auto_data_and_labels(auto_data, f)
        result2 = xval_learning_alg(averaged_perceptron, data, labels, 10)
        result = np.append(result,[result2,features[i][0],features[j][0]])
result = result.reshape(result.shape[0]//3,3)
print("-------------------------------")
print(result)
print("-------------------------------")
print(result[np.argmax(result[:,0])])



# -------------------------------
# [['0.9005128205128207' 'cylinders' 'displacement']
#  ['0.9030128205128207' 'cylinders' 'horsepower']
#  ['0.8953205128205128' 'cylinders' 'weight']
#  ['0.9030128205128207' 'cylinders' 'acceleration']
#  ['0.9055128205128206' 'cylinders' 'origin']
#  ['0.8417948717948718' 'displacement' 'horsepower']
#  ['0.8825000000000001' 'displacement' 'weight']
#  ['0.6456410256410257' 'displacement' 'acceleration']
#  ['0.7985897435897436' 'displacement' 'origin']
#  ['0.8696794871794872' 'horsepower' 'weight']
#  ['0.8392307692307692' 'horsepower' 'acceleration']
#  ['0.821346153846154' 'horsepower' 'origin']
#  ['0.8953846153846154' 'weight' 'acceleration']
#  ['0.8748076923076924' 'weight' 'origin']
#  ['0.7323717948717949' 'acceleration' 'origin']]
# -------------------------------
# ['0.9055128205128206' 'cylinders' 'origin']
