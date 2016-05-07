from os import listdir
from os.path import isfile, join
import dow_jones
import numpy as np
from svm_analysis import read_dictionary
from svm_analysis import intersect

dow_jones_labels = dow_jones.label_nominal(dow_jones.index_changing_rate(dow_jones.read_indices('YAHOO-INDEX_DJI.csv')))

def read_bag_of_word(path):
    word_vector,sentiments = read_dictionary("positive.txt", "negative.txt")
    word_vector=set(word_vector)
    #print(dow_jones_labels)
    files = [f for f in listdir(path) if isfile(join(path, f))]
    bag = {}
    features = {}
    down_count=0
    for value in dow_jones_labels.values():
        if value==-1:
            down_count+=1
    up_count=len(dow_jones_labels)-down_count
    for f in files:
        # print(path + '/' + f)
        date = f.split('.')[0]
        if date not in dow_jones_labels:
            continue
        file = open(path + '/' + f, 'rb')
        content = file.readlines()
        bag = {}
        features[date]=[]
        for line in content:
            line = line.decode("utf-8")
            words = line.split(",")
            if words[0] not in word_vector:
                continue
            features[date].append(words[0])
            # print(words[1])
            if words[0] not in bag:
                if dow_jones_labels[date]<0:
                    bag[words[0]] = {'down': 2, 'up': 1}
                else:
                    bag[words[0]] = {'down': 1, 'up': 2}
            else:
                if dow_jones_labels[date]<0:
                    bag[words[0]]['down'] += 1
                else:
                    bag[words[0]]['up'] += 1
            # print('-------------')
        file.close()
    for word in bag:
        bag[word]['down'] /= float(down_count + 1)
        bag[word]['up'] /= float(up_count + 1)
    return bag, float(down_count)/len(dow_jones_labels), features

def is_down_streaming(content, bag, down_prior_probability):
    down_sum = np.log(down_prior_probability)
    up_sum = np.log(1 - down_prior_probability)
    for word in content:
        # Ignore words that are not encountered in training phase.
        if word not in bag:
            continue
        down_sum += np.log(bag[word]['down'])
        up_sum += np.log(bag[word]['up'])
    print("down={0},up={1}".format(down_sum,up_sum))
    return down_sum > up_sum

if __name__ == "__main__":
    bag,down_prior_probability,features=read_bag_of_word("features")
    print(len(bag))
    for date in features:
        check=is_down_streaming(features[date],bag,down_prior_probability)
        print('{0},{1}'.format(check,dow_jones_labels[date]))