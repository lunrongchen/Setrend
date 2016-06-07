from sklearn import svm
from os import listdir
from os.path import isfile, join
import csv
import dow_jones
import numpy as np
import datetime
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
# This file contains feature of word vector.
# Every dimension correspond to the frequency of word times sentiments defined in the sentimental dictionary.
# Make sure you have run preprocessing.py, dictionary_separator.py before this analysis.
# See function create features.


def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))


def read_dictionary(pos, neg):
    "Read dictionaries. Return a vector of words and a dictionary of sentiments."
    dict = []
    sentiments = {}
    try:
        file1 = open(pos)
        reader = csv.reader(file1, delimiter=',')
        for line in reader:
            dict.append(line[0])
            sentiments[line[0]] = {'sentiment':int(line[2]),'type':line[1]}
        file1.close()
        file2 = open(neg)
        reader = csv.reader(file2, delimiter=',')
        for line in reader:
            dict.append(line[0])
            sentiments[line[0]] = {'sentiment':int(line[2]),'type':line[1]}
        file2.close()
    except IOError:
        print 'Input file reading failed,'
    return [dict, sentiments]


def read_bag_of_word(path):
    "Read pre-processed data"
    files = [f for f in listdir(path) if isfile(join(path, f))]
    bag = {}
    for f in files:
        # print(path + '/' + f)
        date = f.split('.')[0]
        file = open(path + '/' + f, 'rb')
        content = file.readlines()
        bag[date] = {}
        for line in content:
            line = line.decode("utf-8")
            words = line.split(",")
            # print(words[1])
            bag[date][words[0]] = {'tf-idf': float(words[1]), 'count': int(words[2])}
            # print('-------------')
        file.close()
    return bag


def create_features(bag, sentiments, word_vector):
    "This function creates word vector of words in reviews for every day."
    # Every dimension correspond to the frequency of word times sentiments defined in the sentimental dictionary.
    features = {}
    for date in bag:
        features[date] = []
        print('processing '+date)
        sum=0.
        for word in word_vector:
            if word in bag[date]:
                features[date].append(bag[date][word]['count'])
                sum+=bag[date][word]['count']
            else:
                features[date].append(0)
        for i in range(0,len(features[date])):
            features[date][i]/=sum
    return features

def NN_analysis(features,labels):
    #training svm
    print('start to train neural network')
    print('get dates')
    dates=list(labels.keys())
    dates = [datetime.datetime.strptime(ts, "%Y-%m-%d") for ts in dates]
    dates.sort()
    print(dates)
    x=[]
    y=[]
    print('make vectors')
    for date in dates[1:62]:
        time_stamp=datetime.datetime.strftime(date, "%Y-%m-%d")
        feature=np.array(features[time_stamp])
        for i in range(1, 2):
            temp = date-datetime.timedelta(days=i)
            a=np.array(features[datetime.datetime.strftime(temp, "%Y-%m-%d")])
            feature=np.add(feature,a)
        #print(list(feature))
        x.append(feature)
        y.append(labels[time_stamp])
    print('NN training starts')
    x = np.array(x)
    #x.reshape(-1,1)
    ds = SupervisedDataSet(len(x[0]), 1)
    ds.setField('input', x)
    y=np.array(y).reshape( -1, 1 )
    ds.setField('target', y)
    net = buildNetwork(len(x[0]), 500, 1, bias=True)
    trainer = BackpropTrainer(net, ds)
    trainer.trainUntilConvergence(verbose=True, validationProportion=0.15, maxEpochs=500, continueEpochs=10)
    print ("fit finished")
    #training_indices=np.random.choice(len(dates), len(dates)/10)
    count=0
    for i in range(0,len(x)):
        print("predict={0},actual={1}".format(net.activate(x[i]), y[i]))
        if net.activate(x[i]) * y[i] > 0:
            count += 1
    print("accuracy={0}".format(float(count) / len(labels)))

if __name__ == "__main__":
    #read in pre-processed features
    print('reading preprocessed data')
    bag = read_bag_of_word('features')
    #read in sentimental dictionary
    print('reading dictionary')
    [word_vector, sentiments] = read_dictionary("positive.txt", "negative.txt")
    word_vector=set(word_vector)
    word_set = set()
    # word_vector is the intersection of dictionary and words used in all reviews.
    for date in bag.keys():
        for word in bag[date]:
            if word in word_vector:
                word_set.add(word)
    #extract word vector
    print('extracting features')
    word_vector = list(word_set)
    features=create_features(bag,sentiments,word_vector)
    print('word_vector size is '+str(len(word_vector)))
    #read in dow_jones indices
    labels=dow_jones.index_changing_rate(dow_jones.read_indices('YAHOO-INDEX_DJI_longer.csv'))
    #start training
    NN_analysis(features,labels)
