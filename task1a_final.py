import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

#import the data
data = pd.read_csv('train.csv')
X = data.drop('y', axis=1).to_numpy()
y = data['y'].to_numpy()

l = [0.1,1,10,100,200]

#set up splitting the data into training and testing groups
#last time ultimate pi day happened was in 1592 (:
kf = KFold(n_splits=10, shuffle=True, random_state=1592)
averages = np.zeros(5)

#perform Ridge with cross validation for each lambda
for i in range(5):
    RMSE = 0
    for train_index, test_index in kf.split(X): 
        train_X, train_y = X[train_index], y[train_index]
        test_X, test_y = X[test_index], y[test_index]
        model = Ridge(alpha=l[i],fit_intercept=False).fit(train_X, train_y)
        RMSE = RMSE + mean_squared_error(test_y, model.predict(test_X),squared=False)/10
    averages[i] = RMSE
    print(f"The average RMSE for lambda = {l[i]} is: {averages[i]}")

#write averages of the RMSEs to csv file
submission = pd.Series(averages)
submission.to_csv('submission.csv', index=False, header=False)

