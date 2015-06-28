import math
import numpy
import sys
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor

############################################################
### Model 1: Use the average speed and angle in a window ###
############################################################

class MLPredictor:
    """
    The ML predictor class.

    This predictor takes the training data and a prediction horizon:
        (1) normalize the (x,y) coordinate to [0, 1] scale, with 1 representing
        maximium x or y.
        (2) for each point, train a knn regressor based on preivous x points'
        positions, velocities, and angles. x = 0:61 so that we can predict up
        to 60 future points.
        (3) use the knn regressors to predict future points in testing data
    """

    # initialize the predictor
    def __init__(self, n_neighbors, training_file_path):
        # read training data
        self.prediction_horizon = 60
        self.read_training(training_file_path)
        # train the knn classifier
        self.knn_x = [0] * self.prediction_horizon
        self.knn_y = [0] * self.prediction_horizon
        for i in range(self.prediction_horizon):
            self.knn_x[i] = KNeighborsRegressor(n_neighbors, algorithm='kd_tree')
            self.knn_y[i] = KNeighborsRegressor(n_neighbors, algorithm='kd_tree')
            self.knn_x[i].fit(self.training_features, self.training_labels_x[i])
            self.knn_y[i].fit(self.training_features, self.training_labels_y[i])

    def predict(self, testing_labels, horizon):
        predictions = [[testing_labels[0] + self.knn_x[i].predict(testing_labels)[0],
                        testing_labels[1] + self.knn_y[i].predict(testing_labels)[0]]
                       for i in range(horizon)]
        return self.denormalize(predictions)

    # read training data
    def read_training(self, training_file_path):
        training_data = [[int(x) for x in line.rstrip('\r\n').split(',')] for line in open(training_file_path)]
        self.maxX = -1
        self.minX = sys.maxint
        self.maxY = -1
        self.minY = sys.maxint
        for point in training_data:
            self.maxX = max(point[0], self.maxX)
            self.maxY = max(point[0], self.maxY)
            self.minX = min(point[0], self.minX)
            self.minY = min(point[0], self.minY)
        self.training_data = self.normalize(training_data)
        self.training_features = numpy.array(self.training_data)[0:len(self.training_data)-self.prediction_horizon, 0:4]
        self.training_labels_x = []
        self.training_labels_y = []
        for i in range(self.prediction_horizon):
            self.training_labels_x.append(numpy.array(self.training_data)[i:len(self.training_data)-self.prediction_horizon + i, 0] -
                                          numpy.array(self.training_data)[0:len(self.training_data)-self.prediction_horizon, 0])
            self.training_labels_y.append(numpy.array(self.training_data)[i:len(self.training_data)-self.prediction_horizon + i, 1] -
                                          numpy.array(self.training_data)[0:len(self.training_data)-self.prediction_horizon, 1])

    def read_testing(self, testing_file_path, horizon):
        testing_data = [[int(x) for x in line.rstrip('\r\n').split(',')] for line in open(testing_file_path)]
        self.testing_data = self.normalize(testing_data[:len(testing_data) - horizon])
        self.actual_data = self.normalize(testing_data[-horizon:])

    def denormalize(self, data):
        denorm_data =[]
        for point in data:
            denorm_data.append([int(point[0] * (self.maxX - self.minX) + self.minX),
                                int(point[1] * (self.maxY - self.minY) + self.minY)])
        return denorm_data

    # normalize (x,y) data to [0, 1]^2 domain and compute velocity and angle of each point (compared to
    # previous point
    def normalize(self, data):
        norm_data =[]
        for point in data:
            tmp_point = [-1,-1,-1,-1]
            tmp_point[0] = float(point[0] - self.minX) / (self.maxX - self.minX)
            tmp_point[1] = float(point[1] - self.minY) / (self.maxY - self.minY)
            if len(norm_data) > 0:
                tmp_point[2] = math.sqrt((tmp_point[0] - norm_data[-1][0])**2 + (tmp_point[1] - norm_data[-1][1])**2)
                tmp_point[3] = math.atan2(tmp_point[0] - norm_data[-1][0], tmp_point[1] - norm_data[-1][1])
            norm_data.append(tmp_point)
        return norm_data[1:]

    # The main interface of the class.
    def make_prediction(self, testing_file_path, horizon = 0):
        testing_data = [[int(x) for x in line.rstrip('\r\n').split(',')] for line in open(testing_file_path)]
        testing_data_labels = self.normalize(testing_data[:len(testing_data) - horizon])
        actual_data = testing_data[-horizon:]
        predictions = self.predict(testing_data_labels[-1],self.prediction_horizon)
        return [predictions, actual_data]