from csv import reader


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
    category_values = [row[column] for row in dataset]
    unique = set(category_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    return lookup


def conv_col_to_int(dataset, column, lookup):
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


def calc_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0


def train_weights(train, learn_rate, num_epochs):
    weights = [0.0 for i in range(len(train[0]))]
    for epoch in range(num_epochs):
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            weights[0] = weights[0] + learn_rate * error
            for i in range(len(row) - 1):
                weights[i + 1] = weights[i + 1] + learn_rate * error * row[i]
    return weights


def perceptron(train, test, learn_rate, num_epochs):
    predictions = list()
    weights = train_weights(train, learn_rate, num_epochs)
    for row in test:
        predictions.append(predict(row, weights))
    return predictions


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
    cast_to_float(test_data, 3)
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
    return  train_data, test_data


#train, test = prep_adult_data()
train, test = prep_artificial_data()
l_rate = 0.01
#n_epoch = 4
for i in range(20):
    predictions = perceptron(train, test, l_rate, i+1)
    accuracy = calc_accuracy(test[len(test[0]) - 1], predictions)
    print("Epoch:{}".format(i+1))
    print('Scores: %.3f' % accuracy)

