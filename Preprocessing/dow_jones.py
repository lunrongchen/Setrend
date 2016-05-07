import csv
import numpy as np
import scipy
import string


def read_indices(path):
    "Read Dow Jones index from a local file"
    try:
        file = open(path)
        reader = csv.DictReader(file, delimiter=',')
        result = []
        for line in reader:
            result.append(line)
            # line['Date'].replace('//','-')
        # print(result)
        return result
    except IOError:
        print('Input file reading failed')


def index_changing_rate(indices):
    "Calculate the percentage changes of Dow Jones indices"
    result = {}
    for i in range(0, len(indices) - 1):
        result[indices[i]['Date']] = (float(indices[i]['Adjusted Close']) - float(
            indices[i + 1]['Adjusted Close'])) / float(indices[i + 1]['Adjusted Close'])
    return result


def average_abs_rate(rates):
    "Find average absolute changing rate of Dow Jones"
    absolute = []
    for value in rates.values():
        absolute.append(abs(value))
    return np.mean(absolute)


def label_nominal(rates):
    "Convert Dow Jones changes into nominal variables"
    mid = average_abs_rate(rates)
    result = {}
    for date in rates.keys():
        value = rates[date]
        if abs(value) < mid:
            if value < 0:
                result[date] = -1
            else:
                result[date] = 1
        else:
            if value < 0:
                result[date] = -1
            else:
                result[date] = 1
    return result

if __name__ == "__main__":
    indices=read_indices('YAHOO-INDEX_DJI.csv')
    rates=index_changing_rate(indices)
    labels=label_nominal(rates)
    count=0
    for value in labels.values():
        if value==-1:
            count+=1
    print(count)