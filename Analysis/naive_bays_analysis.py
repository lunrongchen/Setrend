from os import listdir
from os.path import isfile, join
import dow_jones
import numpy as np
import datetime
from dictionary_separator import read_dictionary

dow_jones_labels = dow_jones.label_nominal(dow_jones.index_changing_rate(dow_jones.read_indices('YAHOO-INDEX_DJI_longer.csv')))

def read_bag_of_word(path):
    word_vector,sentiments = read_dictionary("positive.txt", "negative.txt")
    word_vector=set(word_vector)
    #print(dow_jones_labels)
    files = [f for f in listdir(path) if isfile(join(path, f))]
    features = {}
    for f in files:
        # print(path + '/' + f)
        date = f.split('.')[0]
        if date not in dow_jones_labels:
            continue
        file = open(path + '/' + f, 'rb')
        content = file.readlines()
        features[date]={}
        for line in content:
            line = line.decode("utf-8")
            words = line.split(",")
            if words[0] not in word_vector:
                continue
            features[date][words[0]]={'count':int(words[2]),'tf-idf':float(words[1])}
            # print(words[1])
            # print('-------------')
        file.close()
    return  features

def naive_bays_training(training_set):
    bag = {}
    down_count=0.
    for value in dow_jones_labels.values():
        if value==-1:
            down_count+=1
    up_count=len(dow_jones_labels)-down_count
    for date in training_set:
        for word in training_set[date]:
            if word not in bag:
                if dow_jones_labels[date] < 0:
                    bag[word] = {'down': 1+1, 'up': 1}
                else:
                    bag[word] = {'down': 1, 'up': 1+1}
            else:
                if dow_jones_labels[date] < 0:
                    bag[word]['down'] += 1
                else:
                    bag[word]['up'] += 1
    for word in bag:
        bag[word]['down'] /= float(down_count + 1)
        bag[word]['up'] /= float(up_count + 1)
        # print("{0},{1}".format(word,bag[word]['up']))
    return bag,down_count/len(dow_jones_labels)

def is_down_streaming(content, bag, down_prior_probability):
    down_sum = np.log(down_prior_probability)
    up_sum = np.log(1 - down_prior_probability)
    for word in content:
        # Ignore words that are not encountered in training phase.
        if word not in bag:
            continue
        down_sum += np.log(bag[word]['down'])
        up_sum += np.log(bag[word]['up'])
    #print("down={0},up={1}".format(down_sum,up_sum))
    if down_sum > up_sum:
        return 1
    else:
        return -1

def cross_validation(features,fold):
    dates=features.keys()
    dates = [datetime.datetime.strptime(ts, "%Y-%m-%d") for ts in dates]
    dates.sort()
    dates = [datetime.datetime.strftime(ts, "%Y-%m-%d") for ts in dates]
    chunk=len(dates)/fold
    average=0
    for i in range(0,fold):
        if (i+1)*chunk<len(dates):
            testing_keys = dates[i*chunk:(i+1)*chunk]
        else:
            testing_keys = dates[i * chunk:len(dates)]
        training_set={}
        testing_set={}
        test_keys_set=set(testing_keys)
        b=False
        size=0
        for date in dates:
            if date in test_keys_set:
                testing_set[date]=features[date]
                b=True
            else:
                training_set[date]=features[date]
                size+=1
        if len(training_set)==0:
            continue
        bag, down_prior_probability = naive_bays_training(training_set)
        #print(bag)
        count=0
        for date in testing_keys:
            prediction = is_down_streaming(testing_set[date],bag,down_prior_probability)
            print("predict={0},actual={1}".format(prediction,dow_jones_labels[date]))
            if prediction * dow_jones_labels[date]>0:
                count += 1
        accuracy=float(count)/len(testing_set)
        print(testing_keys)
        print(len(training_set))
        print("accuracy={0}".format(accuracy))
        average+=accuracy
    print('{0}-fold cross validation accuracy={1}'.format(fold,average/fold))

if __name__ == "__main__":
    features=read_bag_of_word("features")
    count=0
    cross_validation(features, 20)
    #bag,down_prior_probability=naive_bays_training(features)
    #for date in features:
        #check=is_down_streaming(features[date],bag,down_prior_probability)
        #if check==dow_jones_labels[date]:
           # count+=1
        #print('predict={0},actual={1}'.format(check,dow_jones_labels[date]))
    #print("accuracy={0}".format(float(count) / len(dow_jones_labels)))