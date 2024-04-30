# using code_for_hw8_keras.py

#plot_decision('3class', 3, diff=False)
W = np.array([[-0.72990775, 2.3881705, -0.64717734],
              [-1.4610318, 0.3419959, 0.6282539 ]])
W0 = np.array([[-0.10520753],
               [ 0.07852159],
               [ 0.09622384]])
# Use these values to compute each of the z values for the following input points :
pts = np.array([[-1,0], [1,0], [0,-11], [0,1], [-1,-1], [-1,1], [1,1], [1,-1]])

result = np.array([])
for z in pts:
    y = np.argmax(W.T@x+W0.T)
    result = np.append(result, z)
print(result)
