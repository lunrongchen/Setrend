import csv

max_size=5

def read_data(filename,words):
    """Read components of CSV and put them into a list of dictionary"""
    try:
        f = open(filename)
        reader = f.read().splitlines()
        for line in reader:
            #print(line[0])
            words.add(line.lower())
        f.close()
    except IOError:
        print 'Input file reading failed,'
    return words

def contain_financial_topic(sentence,dictionary):
    words=sentence.split(' ')
    for i in range(1,max_size):
        for j in range(1,len(words)):
            temp=j+i
            if temp>len(words):
                break
            test = ' '.join(map(str, words[j:temp]))
            #print(test)
            if test in dictionary:
                #print(test)
                return True
    return False

def build_dictionary():
    dictionary=set()
    read_data('financial_terms.txt',dictionary)
    read_data('financial_dictionary_2.txt',dictionary)
    return dictionary

def main():
    1+1

if __name__ == '__main__':
    main()

