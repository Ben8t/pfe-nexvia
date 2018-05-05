"""
benchmark.py
"""
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dev.model.neural_network_model import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy
import pandas
import time
import seaborn
import matplotlib.pyplot as pyplot


def benchmark_neural_network(x_train, y_train, x_test, y_test):
    """ Compare and test different neural network hyperparameters """
    neural_net_structure = [nn_extractor_structure, nn_expansion_regression_structure]
    number_hidden_layers = [2, 3, 4, 6, 8]
    number_epoch = [10, 100, 250, 500]
    model_result = []
    for structure in neural_net_structure:
        for hidden_layers in number_hidden_layers:
            for epoch in number_epoch:
                start_time = time.time()
                # Build model and prediction
                model = model_neural_network(x_train, y_train, structure, hidden_layers, epoch)
                y_pred = model.predict(x_test)

                # Calcul metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = numpy.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)

                # Save result
                model_name = structure.__name__  # create model name
                time_elapsed = time.time() - start_time
                model_result.append([model_name, hidden_layers, epoch, mae, rmse, r2, time_elapsed])

    benchmark_result = pandas.DataFrame(model_result, columns=["model_name", "hidden_layers", "epoch", "MAE", "RMSE", "R2", "time_elapsed"])
    return benchmark_result


def benchmark_random_forest(x_train, y_train, x_test, y_test):
    """ Compare and test different RandomForest hyperparameters (we can use also GridSearch from scikitlearn) """
    n_estimators = [10, 50, 100, 250, 500, 1000]
    max_features = ["auto", "log2", None]
    max_depth = [2, 3, 5, 10, None]
    model_result = []
    for estimator in n_estimators:
        for feature in max_features:
            for depth in max_depth:
                start_time = time.time()
                print("n_estimators: ", estimator, " - max_features: ", feature, " - max_depth:", depth)
                # Build model and prediction
                model = RandomForestRegressor(n_estimators=estimator, max_features=feature, max_depth=depth)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)

                # Calcul metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = numpy.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)

                # Save result
                model_name = "Random Forest"  # create model name
                time_elapsed = time.time() - start_time
                if(feature == None):
                    feature = "None"
                if(depth == None):
                    depth = "None"
                model_result.append([model_name, estimator, feature, depth, mae, rmse, r2, time_elapsed])

    benchmark_result = pandas.DataFrame(model_result, columns=["model_name", "n_estimators", "max_features", "max_depth", "MAE", "RMSE", "R2", "time_elapsed"])
    return benchmark_result


def benchmark_plot_grid_neural_network(file, save=False, file_name="benchmark_plot_grid_tmp"):
    benchmark_data = pandas.read_csv(file)
    g = seaborn.factorplot(x="epoch", y="MAE",
                           hue="hidden_layers", col="model_name",
                           data=benchmark_data, kind="bar",
                           size=6, aspect=2)
    pyplot.tight_layout()
    if save:
        pyplot.savefig(path + file_name)
        pyplot.close()
    else:
        pyplot.show()


def benchmark_plot_grid_random_forest(file, save=False, file_name="benchmark_plot_grid_tmp"):
    benchmark_data = pandas.read_csv(file)
    g = seaborn.factorplot(x="n_estimators", y="MAE",
                           hue="max_depth", col="max_features",
                           data=benchmark_data, kind="bar",
                           size=4, aspect=.7)
    pyplot.tight_layout()
    if save:
        pyplot.savefig(file_name)
        pyplot.close()
    else:
        pyplot.show()
