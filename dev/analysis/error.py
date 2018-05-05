"""
function to give errors of the model_name:
    * print_score :
        ** print MAE (mean absolute error)
        ** print RMSE (mean quared error)
    * plot_error :
        ** show error graph
    * plot_abs_error:
        ** show absolute error graph with different quantile
    * run_error
        **  if bool_score=True :
                *** print MAE (mean absolute error)
                *** print RMSE (mean quared error)
        **  if bool_plot_error=True :
                *** show error graph
        **  if bool_plot_abs_error=True :
                *** show absolute error graph with different quantile
"""
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as pyplot
import numpy
import seaborn

default_path = "report/img/"


def error_print_score(x_test, y_test, y_pred, model_name="", save=False, path=""):
    """
    Input:
        y_test : real variable
        y_pred : predict variable
    Run:
        print MAE (mean absolute error)
        print RMSE (mean quared error)
    """
    print("modele : ", model_name)
    print("MAE : ", mean_absolute_error(y_test, y_pred))
    print("RMSE : ", numpy.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2 : ", r2_score(y_test, y_pred))


def error_plot(x_test, y_test, y_pred, save=False, path=default_path, model_name="", file_name="error.png"):
    """
    Input:
        y_test : real variable
        y_pred : predict variable
    Run:
        show error graph
    """
    y_error = y_test - y_pred
    pyplot.figure()
    pyplot.plot(y_error, 'bs', marker=".")
    if save:
        pyplot.savefig(path + model_name + "_" + file_name)
        pyplot.close()
    else:
        pyplot.show()


def error_plot_abs(x_test, y_test, y_pred, save=False, path=default_path, model_name="", file_name="error_abs.png"):
    """
    Input:
        y_test : real variable
        y_pred : predict variable
    Run:
        show absolute error graph with different quantile
    """
    y_abs_error = abs(y_test - y_pred)
    pyplot.figure()
    pyplot.plot(y_abs_error, 'bs', marker="o")
    for decile in range(1, 10):
        y_quant = y_abs_error.quantile(decile * 0.1)
        pyplot.plot([y_quant for i in range(min(y_test.index), max(y_test.index) + 1, 1)], 'r', lw=3)
        pyplot.legend(['error', 'quantile from 0.1 to 0.9'])
    if save:
        pyplot.savefig(path + model_name + "_" + file_name)
        pyplot.close()
    else:
        pyplot.show()


def error_input_with_error(x_test, y_test, y_pred, save=False, path=default_path, model_name=""):
    """
    Input:
        x_test : real variable input
        y_test : real variable output
        y_pred : predict variable
    Run:
        show graph error with every input
    """
    y_error = y_test - y_pred
    for name_column in x_test.columns:
        if x_test[name_column].value_counts().shape[0] < 20:
            figure = seaborn.boxplot(x=x_test[name_column], y=y_error)
            figure.set(xlabel=name_column, ylabel='Error')
            pyplot.tight_layout()
        else:
            figure = seaborn.jointplot(x=x_test[name_column], y=y_error, kind="hex", gridsize=30)  # gridsize to change hex size and kind='kde' to density map
            pyplot.tight_layout()
        if save:
            pyplot.savefig(path + model_name + "Residus_" + name_column + ".png")
            pyplot.close()
        else:
            pyplot.show()
