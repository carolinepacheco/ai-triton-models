from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import xgboost as xgb


iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

parameters = {
   "eta": 0.3,
   "objective": "multi:softprob",  # error evaluation for multiclass tasks
   "num_class": 3,  # number of classes to predict
   "max_depth": 3,  # depth of the trees in the boosting process
}

num_round = 20  # the number of training iterations
xgboost = xgb.train(parameters, dtrain, num_round)
xgboost.save_model('xgboost.json')

