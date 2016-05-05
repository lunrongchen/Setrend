import csv

WEAK_NEGATIVE = "weakneg"
STRONG_NEGATIVE = "strongneg"
WEAK_POSITIVE = "weakpos"
STRONG_POSITIVE = "strongpos"
NEUTRAL = "neutral"


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


def opinionfinder_separator(data):
    positive = open("positive.txt", "w")
    negative = open("negative.txt", "w")
    for table in data:
        if table["mpqapolarity"] == WEAK_NEGATIVE:
            negative.write(table["word1"] + "," + table["pos1"] + "," + "-1\n")
        elif table["mpqapolarity"] == STRONG_NEGATIVE:
            negative.write(table["word1"] + "," + table["pos1"] + "," + "-2\n")
        elif table["mpqapolarity"] == STRONG_POSITIVE:
            positive.write(table["word1"] + "," + table["pos1"] + "," + "2\n")
        elif table["mpqapolarity"] == WEAK_POSITIVE:
            positive.write(table["word1"] + "," + table["pos1"] + "," + "1\n")
    positive.close()
    negative.close()


def main():
    data1 = read_data("opinionfinder\\subjclueslen1polar.tff")
    data2 = read_data("opinionfinder\\subjclueslen1polar.tff")
    data = data1 + data2
    opinionfinder_separator(data)


if __name__ == '__main__':
    main()
