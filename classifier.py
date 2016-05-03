import sklearn
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.neural_network import MLPClassifier



# build KNeighborsClassifier
def build_KNN_classifier(data_sets, labels):
    classifier = KNeighborsClassifier()
    classifier.fit(data_sets, labels)
    return classifier


# build LogisticRegression Classifier
def build_LR_classifier(data_sets, labels):
    classifier = LogisticRegression()
    classifier.fit(data_sets, labels)
    return classifier


# build LinearDiscriminantAnalysis 
def build_DA_classifier(data_sets, labels):
    classifier = LinearDiscriminantAnalysis()
    classifier.fit(data_sets, labels)
    return classifier


# build Neural Network Classifier
# def build_NN_classifier(data_sets, labels):
#     classifier = MLPClassifier()
#     classifier.fit(data_sets, labels)
#     return classifier


# build SVM Classifier
def build_SVM_classifier(data_sets, labels):
    classifier = svm.SVC()
    classifier.fit(data_sets, labels)
    return classifier


# build Decision Tree Classifier
def build_DT_classifier(data_sets, labels):
    classifier = DecisionTreeClassifier()
    classifier.fit(data_sets, labels)
    return classifier


# build Bernoulli Naive Bayes Classifier
def build_NB_classifier(data_sets, labels):
    classifier = BernoulliNB()
    classifier.fit(data_sets, labels)
    return classifier

#runs the classifier on test data sets 
def predict_test_data(data_sets, classifier):
    return classifier.predict(data_sets)


# measure the error between predicted and actual
def error_measure(predicted, actual):
    count = 0.0
    for i in range(len(predicted)):
        count += float(abs(predicted[i] - actual[i]))
    return count/float(len(predicted))
    