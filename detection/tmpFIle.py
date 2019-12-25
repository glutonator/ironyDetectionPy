import numpy as np

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

dataX_numpy = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [8, 8], [9, 9]])
dataY_numpy = np.array([0, 0, 1, 1, 1, 0])

X_train_2, X_test, Y_train_2, Y_test = train_test_split(dataX_numpy, dataY_numpy, test_size=0.2, random_state=42)

print(X_train_2)
print(X_test)
print(Y_train_2)
print(Y_test)
print("_______________")

stratSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.5, train_size=0.2, random_state=0)
for train_idx, test_idx in stratSplit.split(dataX_numpy, dataY_numpy):
    print("TRAIN:", train_idx, "TEST:", test_idx)
    X_train, X_test = dataX_numpy[train_idx], dataX_numpy[test_idx]
    print(X_train, '^^^^', X_test)
    print("+++++++++++++++++++++")
    y_train, y_test = dataY_numpy[train_idx], dataY_numpy[test_idx]
    print(y_train, '^^^^', y_test)
    print("+++++++++++++++++++++")
    print("*****************")

#
# sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
# sss.get_n_splits(dataX_numpy, dataY_numpy)
#
# print(sss)
# qqq = sss.split(dataX_numpy, dataY_numpy)
#
# print(dataX_numpy[qqq])
