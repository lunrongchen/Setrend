from NN_analysis import read_dictionary
from NN_analysis import read_bag_of_word
from dictionary_separator import WEAK_NEGATIVE_SCORE
from dictionary_separator import WEAK_POSITIVE_SCORE
from dictionary_separator import STRONG_NEGATIVE_SCORE
from dictionary_separator import STRONG_POSITIVE_SCORE
import dow_jones
import numpy as np
import datetime
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

# This file contains feature of word vector. Each dimension indicates the count for each sentimental categories.
# Make sure you have run preprocessing.py, dictionary_separator.py before this analysis.
dow_jones_labels = dow_jones.index_changing_rate(dow_jones.read_indices('YAHOO-INDEX_DJI_longer.csv'))

def predict_label(bag,word_vector,sentiments,date):
    words=bag[date]
    result=0.
    sum=0
    for word in words:
        if word in word_vector:
            result+=words[word]['count']*sentiments[word]['sentiment']
            sum+=words[word]['count']
    print("{0},{1}".format(result,sum))
    if result/sum>0.08:
        return 1
    else:
        return -1

def create_feature_matrix(bag,sentiments):
    "Extract features from bag of words"
    features=np.empty([len(dow_jones_labels),4])
    feature_dict={}
    target=[]
    i=0
    for date in dow_jones_labels:
        if date not in bag:
            continue
        target.append(dow_jones_labels[date])
        strong_positive_count=0.
        strong_negative_count=0.
        weak_positive_count=0.
        weak_negative_count=0.
        for word in bag[date]:
            if word in sentiments:
                if sentiments[word]['sentiment']==STRONG_POSITIVE_SCORE:
                    strong_positive_count+=bag[date][word]['count']
                else:
                    if sentiments[word]['sentiment'] == STRONG_NEGATIVE_SCORE:
                        strong_negative_count += bag[date][word]['count']
                    else:
                        if sentiments[word]['sentiment'] == WEAK_NEGATIVE_SCORE:
                            weak_negative_count += bag[date][word]['count']
                        else:
                            if sentiments[word]['sentiment'] == WEAK_POSITIVE_SCORE:
                                weak_positive_count += bag[date][word]['count']
        total=strong_positive_count+strong_negative_count+weak_positive_count+weak_negative_count
        features[i,0]=strong_positive_count#/total
        features[i,1]=weak_positive_count#/total
        features[i,2]=strong_negative_count#/total
        features[i,3]=weak_negative_count#/total
        #features[i,0]=(strong_positive_count+weak_positive_count)/total
        #features[i,1]=(strong_negative_count+weak_negative_count)/total
        print("{0},{1},{2},{3},{4}".format(features[i,0],features[i,1],features[i,2],features[i,3],target[i]))
        feature_dict[date]=features[i,0:4]
        i+=1
    # features is an n*4 np array. Each row is a feature of sentiments for that particular day.
    # target is the list of dow jones labels for all dates aligned with features.
    # feature_dict is the dictionary for features using date as key
    return [features,target,feature_dict]

def main():
    #read in pre-processed features
    print('reading preprocessed data')
    bag = read_bag_of_word('features')
    #read in sentimental dictionary
    print('reading dictionary')
    [word_vector, sentiments] = read_dictionary("positive.txt", "negative.txt")
    features,target,features_dict=create_feature_matrix(bag, sentiments)
    # Sort dates in order
    dates=dow_jones_labels.keys()
    dates = [datetime.datetime.strptime(ts, "%Y-%m-%d") for ts in dates]
    dates.sort()
    dates = [datetime.datetime.strftime(ts, "%Y-%m-%d") for ts in dates]

    ds = SupervisedDataSet(4, 1)
    ds.setField('input', features)
    target=np.array(target).reshape( -1, 1 )
    ds.setField('target', target)
    net = buildNetwork(4, 40, 1, bias=True)
    trainer = BackpropTrainer(net, ds)
    trainer.trainUntilConvergence(verbose=True, validationProportion=0.15, maxEpochs=10000, continueEpochs=10)
    count=0
    for i in range(0,len(target)):
        print("predict={0},actual={1}".format(net.activate(features[i]),target[i]))
        if net.activate(features[i])*target[i]>0:
            count+=1
    print("accuracy={0}".format(float(count) / len(dow_jones_labels)))

if __name__ == '__main__':
    main()
