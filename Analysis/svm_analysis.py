from NN_analysis import read_dictionary
from NN_analysis import read_bag_of_word
import dow_jones
from sklearn import svm
from count_sentiments import create_feature_matrix
import datetime

# This file contains feature of word vector. Each dimension indicates the count for each sentimental categories.
# Make sure you have run preprocessing.py, dictionary_separator.py before this analysis.
dow_jones_labels = dow_jones.label_nominal(dow_jones.index_changing_rate(dow_jones.read_indices('YAHOO-INDEX_DJI_longer.csv')))

def cross_validation(fold,dates,features_dict):
    "Cross-alidation using SVM and sentimental features"
    # The most important aspect is that using previous months to predict several days afterwards has accuracy 75%.
    # All folds have accuracy greater than benchmark
    chunk=len(dates)/fold
    average=0
    for i in range(0,fold):
        if (i+1)*chunk<len(dates):
            testing_keys = set(dates[i*chunk:(i+1)*chunk])
        else:
            testing_keys = set(dates[i * chunk:len(dates)])
        training_set=[]
        testing_set=[]
        training_target=[]
        testing_target=[]
        for key in dates:
            if key in testing_keys:
                testing_set.append(features_dict[key])
                testing_target.append(dow_jones_labels[key])
            else:
                training_set.append(features_dict[key])
                training_target.append(dow_jones_labels[key])
        print(training_set)
        clf = svm.SVC()
        clf.fit(training_set, training_target)
        count=0
        for date in testing_keys:
            prediction = clf.predict([features_dict[date]])
            print("predict={0},actual={1}".format(prediction[0],dow_jones_labels[date]))
            if prediction[0] * dow_jones_labels[date]>0:
                count += 1
        accuracy=float(count)/len(testing_set)
        print(testing_keys)
        print("accuracy={0}".format(accuracy))
        average+=accuracy
    print('{0}-fold cross validation accuracy={1}'.format(fold,average/fold))

def main():
    #read in pre-processed features
    print('reading preprocessed data')
    bag = read_bag_of_word('features')
    #read in sentimental dictionary
    print('reading dictionary')
    [word_vector, sentiments] = read_dictionary("positive.txt", "negative.txt")
    features,target,features_dict=create_feature_matrix(bag, sentiments)
    # Sort dates in order
    # Sort dates in order
    dates=dow_jones_labels.keys()
    dates = [datetime.datetime.strptime(ts, "%Y-%m-%d") for ts in dates]
    dates.sort()
    dates = [datetime.datetime.strftime(ts, "%Y-%m-%d") for ts in dates]
    # transform target dow jones index to binary
    cross_validation(10, dates, features_dict)

if __name__ == '__main__':
    main()
