from svm_analysis import read_dictionary
from svm_analysis import read_bag_of_word
from dictionary_separator import WEAK_NEGATIVE_SCORE
from dictionary_separator import WEAK_POSITIVE_SCORE
from dictionary_separator import STRONG_NEGATIVE_SCORE
from dictionary_separator import STRONG_POSITIVE_SCORE
import dow_jones
import numpy as np
from sklearn import svm
import datetime

dow_jones_labels = dow_jones.label_nominal(dow_jones.index_changing_rate(dow_jones.read_indices('YAHOO-INDEX_DJI.csv')))

def predict_label(bag,word_vector,sentiments,date):
    words=bag[date]
    result=0
    for word in words:
        if word in word_vector:
            result+=words[word]['count']*sentiments[word]['sentiment']
    return result>0

def create_feature_matrix(bag,sentiments):
    features=np.empty([len(dow_jones_labels),4])
    feature_dict={}
    target=[]
    i=0
    for date in dow_jones_labels:
        target.append(dow_jones_labels[date])
        strong_positive_count=0
        strong_negative_count=0
        weak_positive_count=0
        weak_negative_count=0
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
        features[i,0]=strong_positive_count
        features[i,1]=weak_positive_count
        features[i,2]=strong_negative_count
        features[i,3]=weak_negative_count
        feature_dict[date]=features[i,0:4]
        i+=1
    return [features,target,feature_dict]



def main():
    #read in pre-processed features
    print('reading preprocessed data')
    bag = read_bag_of_word('features')
    #read in sentimental dictionary
    print('reading dictionary')
    [word_vector, sentiments] = read_dictionary("positive.txt", "negative.txt")
    word_vector=set(word_vector)
    count=0
    features,target,features_dict=create_feature_matrix(bag, sentiments)
    #print(type(features_dict))
    #print(target)
    dates=dow_jones_labels.keys()
    dates = [datetime.datetime.strptime(ts, "%Y-%m-%d") for ts in dates]
    dates.sort()
    dates = [datetime.datetime.strftime(ts, "%Y-%m-%d") for ts in dates]
    fold=5
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
        clf = svm.SVC()
        clf.fit(training_set, training_target)
        count=0
        for date in testing_keys:
            prediction = clf.predict([features_dict[date]])
            if prediction[0] == dow_jones_labels[date]:
                count += 1
        accuracy=float(count)/len(testing_set)
        print(testing_keys)
        print("accuracy={0}".format(accuracy))
        average+=accuracy
    print('{0}-fold cross validation accuracy={1}'.format(fold,average/fold))
    #for date in dow_jones_labels:
        #prediction=predict_label(bag,word_vector,sentiments,date)
       # prediction=clf.predict([dict[date]])
        #if prediction[0]==dow_jones_labels[date]:
         #   count+=1
        #print("predicted={0},actual={1}".format(prediction[0],dow_jones_labels[date]))
    #print("accuracy={0}".format(float(count)/len(dow_jones_labels)))

if __name__ == '__main__':
    main()
