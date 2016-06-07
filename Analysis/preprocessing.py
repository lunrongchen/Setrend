from os import listdir
from os.path import isfile, join
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import unicodedata
import string
import datetime
import threading
import financial_dictionary

stemmer = PorterStemmer()
stop_words=set(stopwords.words('english'))
financial_dict=financial_dictionary.build_dictionary()

def read_reviews(path):
    "Read review text content and arrange them in dates"
    files = [f for f in listdir(path) if isfile(join(path, f))]
    reviews = {}
    lower_bound=datetime.datetime.strptime('2016-01-01', "%Y-%m-%d")
    for f in files:
        print(path + '/' + f)
        file = open(path + '/' + f, 'rb')
        content = file.readlines()
        for line in content:
            line = line.decode("utf-8")
            line = unicodedata.normalize('NFKD', line).encode('ascii', 'ignore')
            words = line.split("\t")
            # print(words[2])
            date = words[2].split('T')[0]
            time_stamp = datetime.datetime.strptime(date, "%Y-%m-%d")
            if time_stamp<lower_bound:
                continue
            words[3] = words[3].translate(None, string.punctuation)
            if not financial_dictionary.contain_financial_topic(words[3],financial_dict):
                #print('filtered out')
                continue
            if date in reviews:
                reviews[date].append(words[3])
            else:
                reviews[date] = [words[3]]
        file.close()
    return reviews


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def bag_of_words_threading(reviews,bag,dates):
    for date in dates:
        dict = {}
        print("bag of words processing " + date)
        for review in reviews[date]:
            lowers = review.lower()
            tokens = tokenize(lowers)
            filtered = [w for w in tokens if not w in stop_words]
            for word in filtered:
                if word in dict:
                    dict[word] += 1
                else:
                    dict[word] = 1
                    # for word in words:
                    #   word = word.lower()
                    #  if word in dict.keys():
                    #     dict[word] += 1
                    # else:
                    #    dict[word] = 1
        bag[date] = dict
    return

def bag_of_words(reviews):
    "Convert reviews to bag of words format"
    dates = reviews.keys()
    bag = {}
    size=len(dates)/8+1
    threads=[]
    for i in range(0,8):
        try:
            if (i+1)*size<=len(dates):
                t=threading.Thread(target=bag_of_words_threading, args=(reviews,bag,dates[i*size:(i+1)*size],))
            else:
                t=threading.Thread(target=bag_of_words_threading, args=(reviews,bag,dates[i*size:len(dates)],))
            threads.append(t)
            t.start()
        except:
            print("Error: unable to start thread "+str(i))
    for t in threads:
        t.join()
    return bag

def tf_idf_threading(table,dates):
    for date in dates:
        corpus = reviews[date]
        if len(corpus) == 1 and len(corpus[0]) == 1:
            continue
        print("TF-IDF processing " + date)

        vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
        vectorizer.fit_transform(corpus)
        idf = vectorizer._tfidf.idf_
        table[date] = dict(zip(vectorizer.get_feature_names(), idf))
    return

def extract_features(reviews,local_copy):
    "Get features of reviews from bag of word in vector format"
    table = {}
    # Compute TF-IDF
    dates = reviews.keys()
    size=len(dates)/8+1
    threads=[]
    for i in range(0,8):
        try:
            if (i+1)*size<=len(dates):
                t=threading.Thread(target=tf_idf_threading, args=(table,dates[i*size:(i+1)*size],))
            else:
                t=threading.Thread(target=tf_idf_threading, args=(table,dates[i*size:len(dates)],))
            threads.append(t)
            t.start()
        except:
            print "Error: unable to start thread"
    for t in threads:
        t.join()
    #print(table)
    # Obtain everyday bag of words
    bag = bag_of_words(reviews)
    # Create word vector from bag of words
    word_set=set()
    for date in bag:
        for word in bag[date]:
            word_set.add(word)
    print(len(word_set))
    if not local_copy:
        return [table,bag,word_set]
    # Write word vector

    # Write bag of word and TF-IDF for everyday
    for date in table:
        f = open('features/' + date + '.csv', 'wb')
        for word in table[date]:
            if word not in bag[date]:
                print("Not part of interest")
                continue
            print(date+' '+word)
            line = word + ',' + str(table[date][word]) + ',' + str(bag[date][word]) + '\n'
            # print(line)
            #line = line.encode("utf-8")
            f.write(line)
        f.close()
    return [table,bag,word_set]


if __name__ == "__main__":
    reviews = read_reviews("Tweets_Data/Data")
    for date in reviews:
        print(reviews[date])
    extract_features(reviews,True)

