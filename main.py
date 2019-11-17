import pandas as pd
import matplotlib.pyplot as plt
import Perceptron as Per


def normalization(iMax, iMin, iVal):
    return (iVal - iMin) / (iMax - iMin)


def normalizeHour(dataframe):
    hourMax = dataframe['Hour'].max()
    hourMin = dataframe['Hour'].min()
    index = 0
    for value in dataframe['Hour']:
        x = normalization(hourMax, hourMin, value)
        dataframe.iat[index, 0] = x
        index = index + 1
    return dataframe


def createScatterGraph(dataframe, day, graphType):
    plt.scatter(dataframe['Hour'], dataframe['Energy Consumption'])
    plt.xlabel('Hour')
    plt.ylabel('Energy Consumption')
    plt.title(graphType + ' Day ' + str(day))
    plt.savefig('ScatterGraphDay' + str(day) + graphType + '.png')
    plt.show()


def createScatterGraph2(dataframe, day, graphType):
    plt.scatter(dataframe['Hour'], dataframe['Prediction'])
    plt.xlabel('Hour')
    plt.ylabel('Energy Consumption')
    plt.title(graphType + ' Day ' + str(day))
    plt.savefig('ScatterGraphDay' + str(day) + graphType + '.png')
    plt.show()


def createLineGraph(dataframe, num):
    plt.plot(dataframe['Hour'], dataframe['Prediction'])
    plt.xlabel('Hour')
    plt.ylabel('Energy Consumption')
    plt.title('Test data graph ' + str(num))
    plt.savefig('LineGraph' + str(num) + '.png')
    plt.show()


def calculateError(model, test, num):
    totalError = 0
    for x in range(0, 15):
        error = model.iat[x, 1] - test.iat[x, 1]
        print('The error for this values is ' + str(error))
        totalError = totalError + abs(error)
    print()
    print("Total Error for " + num + " is " + str(totalError))
    print()


def makeComparisonGraph(testModel, trainedModel, num):
    plt.plot(testModel['Hour'], testModel['Energy Consumption'])
    plt.scatter(trainedModel['Hour'], trainedModel['Prediction'])
    plt.xlabel('Hour')
    plt.ylabel('Energy Consumption')
    plt.title('Comparison Graph Architecture ' + str(num))
    plt.savefig('ComparisonGraph' + str(num))
    plt.show()


def outputModels(day1, day2, day3, test):
    for i in range(0, 3):
        trainingUnit = Per.Perceptron('Linear', i)
        day = str(i+1)
        for j in range(1, 4):
            ite = day + '.' + str(j)
            if j == 1:
                train = day1
            elif j == 2:
                train = day2
            else:
                train = day3
            trainingModel = trainingUnit.train(train)
            createScatterGraph2(trainingModel, ite, 'Training')
            calculateError(trainingModel, test, ite)
        createLineGraph(trainingModel, i+1)
        calculateError(trainingModel, test, 'Day ' + day)
        makeComparisonGraph(test, trainingModel, 'Architecture' + day)


# Import the training data as Pandas Dataframes
trainDay1 = pd.read_csv('Project3_data/train_data_1.txt',
                        sep=',', header=None, names=['Hour', 'Energy Consumption'])
trainDay2 = pd.read_csv('Project3_data/train_data_2.txt',
                        sep=',', header=None, names=['Hour', 'Energy Consumption'])
trainDay3 = pd.read_csv('Project3_data/train_data_3.txt',
                        sep=',', header=None, names=['Hour', 'Energy Consumption'])

# Create line graphs from the dataframes
# Original Graph 1
createScatterGraph(trainDay1, 1, 'Original')

# Original Graph 2
createScatterGraph(trainDay2, 2, 'Original')

# Original Graph 3
createScatterGraph(trainDay3, 3, 'Original')

# Normalize the dataframes
normalDay1 = normalizeHour(trainDay1)
normalDay2 = normalizeHour(trainDay2)
normalDay3 = normalizeHour(trainDay3)

# Create line graphs from the normalized data
# Normal Graph 1
createScatterGraph(normalDay1, 1, 'Normal')

# Normal Graph 2
createScatterGraph(normalDay2, 2, 'Normal')

# Normal Graph 3
createScatterGraph(normalDay3, 3, 'Normal')

# Create a dataframe from the test day
testDay = pd.read_csv('Project3_data/test_data_4.txt',
                      sep=',', header=None, names=['Hour', 'Energy Consumption'])

# Create a line graph from the test day dataframe for comparison
createScatterGraph(testDay, 4, 'Original Test')

# Normalize the test day
normalTest = normalizeHour(testDay)

# Create a line graph from the normalized test day dataframe for comparison
createScatterGraph(normalTest, 4, 'Normal Test')

outputModels(normalDay1, normalDay2, normalDay3, normalTest)

print("End of file")
