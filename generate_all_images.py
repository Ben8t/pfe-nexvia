from os import listdir, path, makedirs
from inspect import getmembers, isfunction
from dev.analysis import analysis as modul_analysis
from dev.analysis import error as modul_error
from dev.model import model as modul_model
from dev.analysis.analysis import *
from dev.analysis.error import *
from dev.model.model import *
from dev.processing import data_processing as modul_process
from dev.processing.load_data import load_data
from dev.processing.data_processing import *
from dev.model.neural_network_model import *

#############################################################################
#                                                                           #
#             function to search for different possibilities                #
#                                                                           #
#############################################################################


def create_choice_list(modul_name, key_word):
    """
    return a function list where we find functions of modul whose name includes "keyword"
    """
    functions_list = [getattr(modul_name, o[0]) for o in getmembers(modul_name) if isfunction(o[1])]
    filter_functions_list = list(filter(lambda function: key_word in function.__name__, functions_list))
    return filter_functions_list


#############################################################################
#                                                                           #
#                  loading data and cleaning data                           #
#                                                                           #
#############################################################################

data = load_data()
cleaned_data = cleaning_data(data)

#############################################################################
#                                                                           #
#             function to generate analysis images                          #
#                                                                           #
#############################################################################

choice_analysis = create_choice_list(modul_analysis, "analysis_")
choice_error = create_choice_list(modul_error, "error_")
for analysis_function in choice_analysis:
    analysis_function(cleaned_data, save=True)
print("analysis done")

#############################################################################
#                                                                           #
#             separation processing according to dummies                    #
#                                                                           #
#############################################################################

processing_with_dummies = [processing_full, processing_less_feature, processing_full_polynomial_feature]
processing_without_dummies = [processing_label_encoder, processing_change_interval]
processing_all = processing_with_dummies + processing_without_dummies

#############################################################################
#                                                                           #
#               separation model according to dummies                       #
#                                                                           #
#############################################################################

model_with_dummies = [model_homeType_regression, model_homeType_random_forest]
model_without_dummies = [model_multidiscret_regression, model_multidiscret_random_forest]
model_sucks_dummies = [model_regression, model_decision_tree, model_random_forest]


################################
#       model_accept_dummies   #
################################

for process_function in processing_with_dummies:
    processed_data = process_function(cleaned_data)
    x_train, y_train, x_test, y_test = split_dataset(processed_data, 0.3, "output-cumulatedCostsBuy_homeAcquisitionCosts_1")
    model_accept_dummies = model_with_dummies + model_sucks_dummies
    for model_function in model_accept_dummies:
        model = model_function(x_train, y_train)
        y_pred = model.predict(x_test)
        model_name = model_function.__name__ + "_" + process_function.__name__ + "_"
        for error_function in choice_error:
            if not path.exists("report/img/" + model_name):
                makedirs("report/img/" + model_name)
            error_function(x_test, y_test, y_pred, save=True, path="report/img/" + model_name + "/", model_name=model_name)
print("model_dummies done")
################################
#    model_refuse_dummies      #
################################

for process_function in processing_without_dummies:
    processed_data = process_function(cleaned_data)
    x_train, y_train, x_test, y_test = split_dataset(processed_data, 0.3, "output-cumulatedCostsBuy_homeAcquisitionCosts_1")
    model_refuse_dummies = model_without_dummies + model_sucks_dummies
    for model_function in model_refuse_dummies:
        model = model_function(x_train, y_train)
        y_pred = model.predict(x_test)
        model_name = model_function.__name__ + "_" + process_function.__name__ + "_"
        for error_function in choice_error:
            if not path.exists("report/img/" + model_name):
                makedirs("report/img/" + model_name)
            error_function(x_test, y_test, y_pred, save=True, path="report/img/" + model_name + "/", model_name=model_name)
print("model without done")


#############################################################################
#                                                                           #
#                           neural network                                  #
#                                                                           #
#############################################################################

list_structure = [nn_two_hidden_layers_structure]
for process_function in processing_with_dummies:
    processed_data = process_function(cleaned_data)
    x_train, y_train, x_test, y_test = split_dataset(processed_data, 0.3, "output-cumulatedCostsBuy_homeAcquisitionCosts_1")
    for structure in list_structure:
        pipe = model_neural_network(x_train, y_train, structure)
        y_pred = pipe.predict(X_test)
        model_name = structure.__name__ + "_" + process_function.__name__ + "_"
        for error_function in choice_error:
            if not path.exists("report/img/" + model_name):
                makedirs("report/img/" + model_name)
            error_function(x_test, y_test, y_pred, save=True, path="report/img/" + model_name + "/", model_name=model_name)
print("neural done")
