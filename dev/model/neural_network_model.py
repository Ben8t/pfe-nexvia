"""
neural_network_model.py
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import load_model
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy
import pandas
import time


def model_neural_network(x_train, y_train, model_structure, number_hidden_layers, nb_epochs=1, save=False):
    """
    Neural networks model, encapsulated in scikit Pipeline
    """

    # Pipeline
    pipe = Pipeline([
        ('normalise', StandardScaler()),  # normalize data for neural net
        ('model', KerasRegressor(build_fn=model_structure,
                                 number_hidden_layers=number_hidden_layers,
                                 input_dim=x_train.shape[1],
                                 epochs=nb_epochs,
                                 batch_size=16))])  # Use Keras wrappers for the Scikit-Learn API

    pipe.fit(x_train, y_train)

    if save:
        save_pipeline_keras_model(pipe, model_structure.__name__, "pipeline")

    return pipe


def nn_two_hidden_layers_structure(input_dim):
    """ Simple neural net structure """
    model = Sequential([
        Dense(25, input_dim=input_dim, activation='relu'),
        Dense(20, activation='relu'),
        Dense(1, activation='linear')])

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
    return model


def nn_expansion_regression_structure(input_dim, number_hidden_layers):
    """ Define neural network structure : expansion-regression refers to the shape of the neural net"""
    mid_n = int(number_hidden_layers / 2)  # middle index of the structure
    layers_size = [int(input_dim * 1.5**i) for i in range(mid_n + 1)]  # size of each layers, proportional to input shape

    # Input layer
    layers_sequence = [Dense(layers_size[1], input_dim=input_dim, activation='relu')]

    # Check if with there are pair/impair number of layers
    if (number_hidden_layers % 2) == 0:
        adjust_size = 0
    else:
        adjust_size = 1

    # Expansion phase
    for step_layer in range(2, mid_n + adjust_size):
        layers_sequence.append(Dense(layers_size[step_layer], activation='relu'))

    # Regression phase
    for step_layer in range(mid_n, -1, -1):
        layers_sequence.append(Dense(layers_size[step_layer], activation='relu'))

    # Output layer
    layers_sequence.append(Dense(1, activation='linear'))
    model = Sequential(layers_sequence)

    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
    return model


def nn_extractor_structure(input_dim, number_hidden_layers):
    """ Define neural network structure : extractor refers to the shape of the neural net"""
    layers_size = [int(input_dim * 1.5**i) for i in range(1, number_hidden_layers + 1)]  # size of each layers, proportional to input shape

    # Input layer
    layers_sequence = [Dense(layers_size[0], input_dim=input_dim, activation='relu')]

    # Extractor phase
    for step_layer in range(1, number_hidden_layers):
        layers_sequence.append(Dense(layers_size[step_layer], activation='relu'))

    # Output layer
    layers_sequence.append(Dense(1, activation='linear'))

    model = Sequential(layers_sequence)

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
    model.summary()
    return model
