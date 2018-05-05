"""
functions to analyze the data-frame
    heatmap : correlation map
    graph_home_price : show graph with homePrice and
                       the output according to homeType
    graph_every_input_with_output : how every input with the output in a graph
"""
# variable analysis
# import pandas
import seaborn
import pandas
import matplotlib.pyplot as pyplot
from dev.processing.data_processing import processing_full
default_path = "report/img/"


def analysis_heatmap(cleaned_data, save=False, path=default_path, file_name="heatmap.png"):
    """
    cleaned_data is a data-frame
    show a map with correlation
    """
    processed_data = processing_full(cleaned_data)
    pyplot.figure(figsize=(15, 10))
    correlation = processed_data.corr()
    seaborn.heatmap(correlation, annot=True, annot_kws={"size": 10}, fmt=".2f")
    pyplot.tight_layout()
    if save:
        pyplot.savefig(path + file_name)
        pyplot.close()
    else:
        pyplot.show()


def analysis_graph_home_price(cleaned_data, save=False, path=default_path, file_name="graph_home_price"):
    """
    cleaned_data is a data-frame
    show graph with homePrice and the output according to homeType
    """
    input_variable = "input-homePrice"
    separation_variable = "input-homeType"
    output_variable = "output-cumulatedCostsBuy_homeAcquisitionCosts_1"
    figure = seaborn.lmplot(input_variable, output_variable, data=cleaned_data, hue=separation_variable, fit_reg=False)
    figure.set(xlabel=input_variable, ylabel=output_variable)
    pyplot.tight_layout()
    if save:
        pyplot.savefig(path + file_name)
        pyplot.close()
    else:
        pyplot.show()


def analysis_graph_home_price2(cleaned_data, save=False, path=default_path, file_name="graph_home_price2"):
    """
    cleaned_data is a data-frame
    show graph with homePrice and the output according to homeType
    """
    new_data = cleaned_data.copy()
    new_data["marital-homeType"] = new_data['input-maritalStatus'] + "/" + new_data['input-homeType']
    input_variable = "input-homePrice"
    separation_variable = "marital-homeType"
    output_variable = "output-cumulatedCostsBuy_homeAcquisitionCosts_1"
    figure = seaborn.lmplot(input_variable, output_variable, data=new_data, hue=separation_variable, fit_reg=False)
    figure.set(xlabel=input_variable, ylabel=output_variable)
    pyplot.tight_layout()
    if save:
        pyplot.savefig(path + file_name)
        pyplot.close()
    else:
        pyplot.show()


def analysis_graph_every_input_with_output(cleaned_data, save=False, path=default_path):
    """
    cleaned_data is a data-frame
    show every input with the output in a graph
    """
    output = "output-cumulatedCostsBuy_homeAcquisitionCosts_1"
    for name_column in cleaned_data.columns:
        if name_column != output:
            if cleaned_data[name_column].value_counts().shape[0] < 20:
                figure = seaborn.boxplot(x=name_column, y=output, data=cleaned_data)
                figure.set(xlabel=name_column, ylabel=output)
                pyplot.tight_layout()
            else:
                figure = seaborn.jointplot(x=name_column, y=output, data=cleaned_data, kind="hex", gridsize=30)  # gridsize to change hex size and kind='kde' to density map
                pyplot.tight_layout()
            if save:
                pyplot.savefig(path + name_column + ".png")
                pyplot.close()
            else:
                pyplot.show()


def analysis_homeType_homePrice(cleaned_data, save=False, path=default_path, file_name="homeType_homePrice.png"):
    """
    plot homePrice for each home type
    """
    figure = seaborn.boxplot(x="input-homeType", y="input-homePrice", data=cleaned_data)
    figure.set(xlabel='homeType', ylabel='homePrice')
    pyplot.tight_layout()
    if save:
        pyplot.savefig(path + file_name)
        pyplot.close()
    else:
        pyplot.show()


def analysis_histogram(cleaned_data, save=False, path=default_path, file_name="histogram.png"):
    """
    plot each features distributions
    """
    cleaned_data.hist(bins=20, edgecolor='black', linewidth=1.0, xlabelsize=8, ylabelsize=8, grid=False)
    pyplot.tight_layout()
    if save:
        pyplot.savefig(path + file_name)
        pyplot.close()
    else:
        pyplot.show()


def feature_importance_barplot(model, data, save=False, path=default_path, file_name="feature_importance_barplot"):
    """
    plot feature importance from a tree model (sklearn.tree)
    data is just to get columns names
    """
    result_decision_tree = pandas.DataFrame(
        {"features": list(data.columns),
         "importance": list(model.feature_importances_)})
    print(result_decision_tree)
    figure = seaborn.barplot(x="importance", y="features", data=result_decision_tree)
    pyplot.tight_layout()
    if save:
        pyplot.savefig(path + file_name)
        pyplot.close()
    else:
        pyplot.show()
