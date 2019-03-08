from csv import reader
import math
import operator



def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def cast_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


def categorical_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    return lookup


def conv_col_to_int(dataset, column, lookup):
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    class_votes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sortedVotes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0



def prep_adult_data():
    filename = 'adult.data'
    dataset = load_csv(filename)
    cast_to_float(dataset, 0)
    look_1 = categorical_to_int(dataset, 1)
    conv_col_to_int(dataset, 1, look_1)
    cast_to_float(dataset, 2)
    look_2 = categorical_to_int(dataset, 3)
    conv_col_to_int(dataset, 3, look_2)
    cast_to_float(dataset, 4)
    look_3 = categorical_to_int(dataset, 5)
    conv_col_to_int(dataset, 5, look_3)
    look_4 = categorical_to_int(dataset, 6)
    conv_col_to_int(dataset, 6, look_4)
    look_5 = categorical_to_int(dataset, 7)
    conv_col_to_int(dataset, 7, look_5)
    look_6 = categorical_to_int(dataset, 8)
    conv_col_to_int(dataset, 8, look_6)
    look_7 = categorical_to_int(dataset, 9)
    conv_col_to_int(dataset, 9, look_7)
    cast_to_float(dataset, 10)
    cast_to_float(dataset, 11)
    cast_to_float(dataset, 12)
    look_8 = categorical_to_int(dataset, 13)
    conv_col_to_int(dataset, 13, look_8)
    look_9 = categorical_to_int(dataset, 14)
    conv_col_to_int(dataset, 14, look_9)

    test_data = load_csv('adult.test')
    for row in test_data:
        row[-1] = row[-1][:-1]
    cast_to_float(test_data, 0)
    conv_col_to_int(test_data, 1, look_1)
    cast_to_float(test_data, 2)
    conv_col_to_int(test_data, 3, look_2)
    cast_to_float(test_data, 4)
    conv_col_to_int(test_data, 5, look_3)
    conv_col_to_int(test_data, 6, look_4)
    conv_col_to_int(test_data, 7, look_5)
    conv_col_to_int(test_data, 8, look_6)
    conv_col_to_int(test_data, 9, look_7)
    cast_to_float(test_data, 10)
    cast_to_float(test_data, 11)
    cast_to_float(test_data, 12)
    conv_col_to_int(test_data, 13, look_8)
    conv_col_to_int(test_data, 14, look_9)
    return dataset, test_data


def prep_artificial_data():
    train_data = load_csv('artificial_with_noise_train.csv')
    look = []
    for i in range(4):
        look.append(categorical_to_int(train_data, i))
    test_data = load_csv('artificial_with_noise_test.csv')
    for i in range(4):
        conv_col_to_int(train_data, i, look[i])
        conv_col_to_int(test_data, i, look[i])
    return train_data, test_data


#train, test = prep_adult_data()
train, test = prep_artificial_data()
predictions=[]
k = 5
for x in range(len(test)):
    neighbors = getNeighbors(train, test[x], k)
    result = getResponse(neighbors)
    predictions.append(result)
accuracy = getAccuracy(test, predictions)
print('Accuracy: ' + repr(accuracy) + '%')