import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# all data for training
def kNNTrain1():
    raw_data_X1 = [[3.393533211, 2.331273381], [3.110073483, 1.781539638],
                   [1.343808831, 3.368360954], [3.582294042, 4.679179110],
                   [2.280362439, 2.866990263], [7.423436942, 4.696522875],
                   [5.745051997, 3.533989803], [9.172168622, 2.511101045],
                   [7.792783481, 3.424088941], [7.939820817, 0.791637231]]
    raw_data_y1 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    X_train1 = np.array(raw_data_X1)
    y_train1 = np.array(raw_data_y1)
    x1 = np.array([8.093607318, 3.365731514])
    kNN_classifier1 = KNeighborsClassifier(n_neighbors=6, n_jobs=-1)
    kNN_classifier1.fit(X_train1, y_train1)
    X_predict1 = x1.reshape(1, -1)
    y_predict1 = kNN_classifier1.predict(X_predict1)
    plt.scatter(X_train1[y_train1 == 0, 0],
                X_train1[y_train1 == 0, 1],
                color='b')
    plt.scatter(X_train1[y_train1 == 1, 0],
                X_train1[y_train1 == 1, 1],
                color='r')
    plt.scatter(X_predict1[0, 0], X_predict1[0, 1], color='g', marker='*')
    plt.show()
    return y_predict1[0]


# data for training and testing
def kNNTrain2():
    iris = datasets.load_iris()
    iris.keys()
    X2 = iris.data
    y2 = iris.target
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,
                                                            y2,
                                                            test_size=0.9,
                                                            random_state=666)
    x2 = np.array([8.093607318, 3.365731514, 1, 0.2])
    kNN_classifier2 = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    kNN_classifier2.fit(X_train2, y_train2)
    X_predict2 = x2.reshape(1, -1)
    y_predict2 = kNN_classifier2.predict(X_test2)
    print("The score for prediction is:", accuracy_score(y_test2, y_predict2))
    plt.scatter(X_train2[y_train2 == 0, 0],
                X_train2[y_train2 == 0, 1],
                color='b')
    plt.scatter(X_train2[y_train2 == 1, 0],
                X_train2[y_train2 == 1, 1],
                color='r')
    plt.scatter(X_predict2[0, 0], X_predict2[0, 1], color='g', marker='+')
    #plt.show()
    return y_predict2[0]


print("Below is the prediction from kNNTrain1")
print("The prediction is:", kNNTrain1())
print("Below is the prediction from kNNTrain2")
print("The prediction is:", kNNTrain2())
