from csv import reader
import math


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



def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def deviation(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attribute), deviation(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


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
summaries = summarizeByClass(train)
predictions = getPredictions(summaries, test)
accuracy = getAccuracy(test, predictions)
print('Accuracy: {0}%'.format(accuracy))


