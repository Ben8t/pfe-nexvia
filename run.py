"""
give the choice to the user to make all function in the project:
https://goo.gl/J6C7VP
"""

from os import listdir
from inspect import getmembers, isfunction
from dev.analysis import analysis as modul_analysis
from dev.analysis import error as modul_error
from dev.model import model as modul_model
from dev.processing import data_processing as modul_processing
from dev.analysis.analysis import *
from dev.analysis.error import *
from dev.model.model import *
from dev.processing.data_processing import *
from dev.processing.load_data import load_data_shuffle, load_data

#boolean to quit a while loop:
dictionary_boolean = {'exit': False,
                      'new_data' : False,
                      'new_processing' : False,
                      'new_work_on_same_data' : False,
                      'new_model' : False,
                      'new_save_image' : False
                     }
#list of dictionary to rank the dictionary key
list_boolean = ['exit', 'new_data', 'new_processing', 'new_work_on_same_data', 'new_model','new_save_image']

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

def create_choice_list_data():
    """
    return a zip file name list which are in the data folder
    """
    fichiers_list = listdir('data/')
    filter_fichiers_list = list(filter(lambda fichier: '.zip' in fichier, fichiers_list))
    return filter_fichiers_list

#############################################################################
#                                                                           #
#                       list giving user's choices                          #
#                                                                           #
#############################################################################

choice_data = create_choice_list_data()
#still missing the data with the number chosen by the user. Completed in run function
choice_analysis = create_choice_list(modul_analysis, "analysis_")
choice_model = create_choice_list(modul_model, "model_")
choice_error = create_choice_list(modul_error, "error_")
choice_processing = create_choice_list(modul_processing, "processing_")
choice_work_to_do = None #completed after functions that do the work

#############################################################################
#                                                                           #
#                  functions to get the  user's choices                     #
#                                                                           #
#############################################################################

def create_text_input(choice_list, number_option):
    """
    Rreturn a text specifing the different options of the user (depend of the choice_list)
    """
    input_choix = ['-6 pour retourner à la visualisation des erreur du modele\n',
                   '-5 pour avoir un nouveau model\n',
                   '-4 pour avoir un nouveau processing\n',
                   '-3 pour une nouvelle tache avec les memes donnes\n',
                   '-2 pour changer les donnees\n',
                   '-1 pour quitter\n']
    input_text = "Voici vos différents choix :\n"
    for add_line in range(number_option, 0, -1):
        input_text = input_text + input_choix[-add_line]
    if number_option == 1:
        for index, item in enumerate(choice_list):
            input_text = input_text +"{}".format(index)+" pour "+item+"\n"
        input_text = input_text +\
                        "n <= 20 000 : pour des donnees aleatoires de taille n \n"
    else:
        for index, item in enumerate(choice_list):
            input_text = input_text +"{}".format(index)+" pour "+item.__name__+"\n"
    return input_text

def get_user_choice(minimum, maximum, input_text):
    user_choice = maximum+1
    while not minimum <= user_choice <= maximum:
        try:
            user_choice = int(input(input_text))
            if not minimum <= user_choice <= maximum:
                print("\n Entrée non valide \n")
        except ValueError:
            print("\n Entrée non valide \n")
    return user_choice

def get_choice(choice_list, number_option):
    input_text = create_text_input(choice_list, number_option)
    if number_option == 1:
        maximum = 20000
    else:
        maximum = len(choice_list) - 1
    user_choice = get_user_choice(-number_option, maximum, input_text)
    print("\n")
    return user_choice

#############################################################################
#                                                                           #
#                    functions that manage the booleens                     #
#                                                                           #
#############################################################################

def alter_dictionary(threshold, boolean):
    for index, name_boolean in enumerate(list_boolean):
        if index >= threshold:
            dictionary_boolean[name_boolean] = boolean

def settings_boolean(switch):
    if switch > 0:
        alter_dictionary(switch, False)
    else:
        threshold = abs(switch)-1
        alter_dictionary(threshold, True)

#############################################################################
#                                                                           #
#                         functions that do the work                        #
#                                                                           #
#############################################################################
def save_errors(model_name, x_test, y_test, y_pred):
    print("\nchoix de sauvegarde\nmodel",model_name)
    while not dictionary_boolean['new_save_image']:
        settings_boolean(6)
        user_error_choice = get_choice(choice_error, 6)
        if user_error_choice < 0:
            settings_boolean(user_error_choice)
        else:
            choice_error[user_error_choice](x_test, y_test, y_pred,save=True, model_name=model_name)

def analysis(work_on_cleaned_data):
    user_analysis_choice = get_choice(choice_analysis, 3)
    if user_analysis_choice < 0:
        settings_boolean(user_analysis_choice)
    else:
        choice_analysis[user_analysis_choice](work_on_cleaned_data)

def model(work_on_cleaned_data):
    user_processed_choice = get_choice(choice_processing, 3)
    if user_processed_choice < 0:
        settings_boolean(user_processed_choice)
    else:
        work_on_processed_data = choice_processing[user_processed_choice](work_on_cleaned_data)
        x_train, y_train, x_test, y_test = split_dataset(work_on_processed_data, 0.3, "output-cumulatedCostsBuy_homeAcquisitionCosts_1")
    while not dictionary_boolean['new_processing']:
        settings_boolean(4)    
        user_model_choice = get_choice(choice_model, 4)
        if user_model_choice < 0:
            settings_boolean(user_model_choice)
        else:
            model = choice_model[user_model_choice](x_train, y_train)
            y_pred = model.predict(x_test)            
            model_name = choice_model[user_model_choice].__name__
            while not dictionary_boolean['new_model']:
                print("\n", model_name)
                settings_boolean(5)
                user_error_choice = get_choice(choice_error+[save_errors], 5)
                if user_error_choice < 0:
                    settings_boolean(user_error_choice)
                else:
                    if user_error_choice == len(choice_error):
                        save_errors(model_name, x_test, y_test, y_pred)
                    else:
                        choice_error[user_error_choice](x_test, y_test, y_pred)

def save_analysis(work_on_cleaned_data):
    print("\nSauvegarder des images")
    user_analysis_choice = get_choice(choice_analysis, 3)
    if user_analysis_choice < 0:
        settings_boolean(user_analysis_choice)
    else:
        choice_save_analysis[user_analysis_choice](work_on_cleaned_data, save=True)
        print("work done")

#############################################################################
#                                                                           #
#                               main function                               #
#                                                                           #
#############################################################################
choice_work_to_do = [analysis, model, save_analysis]
def main():
    while not dictionary_boolean['exit']:
        settings_boolean(1)
        user_data = get_choice(choice_data, 1)
        if user_data < 0:
            settings_boolean(user_data)
        else:
            if -1 < user_data < len(choice_data):
                raw_data = load_data("data/"+choice_data[user_data])
            else:
                raw_data = load_data_shuffle(user_data)
            cleaned_data = cleaning_data(raw_data)
            while not dictionary_boolean['new_data']:
                settings_boolean(2)
                user_choice_work_to_do = get_choice(choice_work_to_do, 2)
                if user_choice_work_to_do < 0:
                    settings_boolean(user_choice_work_to_do)
                else:
                    while not dictionary_boolean['new_work_on_same_data']:
                        settings_boolean(3)
                        choice_work_to_do[user_choice_work_to_do](cleaned_data)

#############################################################################
#                                                                           #
#                                   the run                                 #
#                                                                           #
#############################################################################
main()
