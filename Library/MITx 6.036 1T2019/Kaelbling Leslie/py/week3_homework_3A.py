# Enter a list of 6 integers indicating the number of polynomial features 
# of degrees [1, 10, 20, 30, 40, 50] for a 2-dimensional feature vector.

# see https://stackoverflow.com/questions/31290976/sklearn-how-to-get-coefficients-of-polynomial-features

from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np

data = pd.DataFrame.from_dict({
    'x': np.random.randint(low=1, high=1, size=1),
    'y': np.random.randint(low=1, high=1, size=1),
})

result = np.array([])
degrees = [1, 10, 20, 30, 40, 50]
for d in degrees:
    p = PolynomialFeatures(degree=d).fit(data)
    result = np.append(result ,np.array(p.get_feature_names(data.columns)).shape)

print(result)

# (3,66,231,496,861,1326)
