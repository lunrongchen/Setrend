import csv

WEAK_NEGATIVE = "weakneg"
STRONG_NEGATIVE = "strongneg"
WEAK_POSITIVE = "weakpos"
STRONG_POSITIVE = "strongpos"
NEUTRAL = "neutral"
STRONG_NEGATIVE_SCORE=-1
STRONG_POSITIVE_SCORE=2
WEAK_NEGATIVE_SCORE=-1
WEAK_POSITIVE_SCORE=2


def read_data(filename):
    """Read components of CSV and put them into a list of dictionary"""
    data = []
    try:
        f = open(filename)
        reader = csv.reader(f, delimiter=' ')
        for line in reader:
            if line[0][0] == "#":
                continue
            table = dict()
            for entry in line:
                temp = entry.split("=")
                if len(temp) > 1:
                    table[temp[0]] = temp[1]
            data.append(table)
            # print table
    except IOError:
        print 'Input file reading failed,'
    return data

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


def opinionfinder_separator(data):
    positive = open("positive.txt", "w")
    negative = open("negative.txt", "w")
    for table in data:
        if table["mpqapolarity"] == WEAK_NEGATIVE:
            negative.write(table["word1"] + "," + table["pos1"] + "," + str(WEAK_NEGATIVE_SCORE)+ "\n")
        elif table["mpqapolarity"] == STRONG_NEGATIVE:
            negative.write(table["word1"] + "," + table["pos1"] + "," + str(STRONG_NEGATIVE_SCORE) + "\n")
        elif table["mpqapolarity"] == STRONG_POSITIVE:
            positive.write(table["word1"] + "," + table["pos1"] + "," + str(STRONG_POSITIVE_SCORE) + "\n")
        elif table["mpqapolarity"] == WEAK_POSITIVE:
            positive.write(table["word1"] + "," + table["pos1"] + "," + str(WEAK_POSITIVE_SCORE) + "\n")
    positive.close()
    negative.close()


def main():
    data1 = read_data("opinionfinder/subjclueslen1polar.tff")
    data2 = read_data("opinionfinder/subjclueslen1polar.tff")
    data = data1 + data2
    opinionfinder_separator(data)


if __name__ == '__main__':
    main()
