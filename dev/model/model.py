"""
model definition
"""
from dev.processing.data_processing import split_dataset
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

import numpy
import pandas


class DoubleHomeType(BaseEstimator, RegressorMixin):
    """
    Distinct two models according to homeType feature
    """

    def __init__(self, model=None):
        """
        model : a fitted model (example : LinearRegression().fit(x_train, y_train))
        """
        self.model = model
        pass

    def fit(self, x_train, y_train):
        """
        fit two models : model_new for homeType == new and model_existing for homeType == existing
        """
        output = "output-cumulatedCostsBuy_homeAcquisitionCosts_1"
        train = pandas.concat([x_train, y_train], axis=1)
        train_new = train[train["input-homeType_new"] == 1]
        train_existing = train[train["input-homeType_existing"] == 1]
        x_train_new = train_new.drop(output, axis=1)
        y_train_new = train_new[output]
        x_train_existing = train_existing.drop(output, axis=1)
        y_train_existing = train_existing[output]
        self.model_new = self.model(x_train_new, y_train_new)
        self.model_existing = self.model(x_train_existing, y_train_existing)
        return self

    def predict(self, X):
        """
        predict with an indicator function (ie. prediction is model_new if homeType == new and model_existing if homeType == existing)
        """
        return X['input-homeType_new'] * self.model_new.predict(X) + X['input-homeType_existing'] * self.model_existing.predict(X)


class SplitDiscreteFeatureModel(BaseEstimator, RegressorMixin):

    def __init__(self, model, list_discret_variable):
        self.model = model
        self.list_discret_variable = list_discret_variable
        pass

    def fit(self, x_train, y_train):
        self.list_model_valeur = []
        self.fit_recursive(x_train, y_train, [])
        return self

    def fit_recursive(self, x_train, y_train, list_value):
        if len(self.list_discret_variable) == len(list_value):
            output = "output-cumulatedCostsBuy_homeAcquisitionCosts_1"
            train = pandas.concat([x_train, y_train], axis=1)
            for index in range(len(list_value)):
                train = train[train[self.list_discret_variable[index]] == list_value[index]]
            # cas problématique : scchéma de variable possible qui n'apparait pas assez pour l'apprentissage du sous model ou pire jamais : estimation = 0
            if train.shape[0] > 0:
                # seperation des lignes suivant les variables concernés
                x_train_new = train.drop(output, axis=1)
                y_train_new = train[output]
                # entrainement du sous model
                model = self.model(x_train_new, y_train_new)
                self.list_model_valeur.append((model, list_value))
        else:
            # on mets les chaines de valeurs possibles
            for value in list(x_train[self.list_discret_variable[len(list_value)]].value_counts().index):
                new_list = list_value + [value]
                self.fit_recursive(x_train, y_train, new_list)

    def predict(self, x_test):
        n_row = len(x_test)
        n_discret_variable = len(self.list_discret_variable)
        result = [0 for i in range(n_row)]
        for model, list_value in self.list_model_valeur:
            result_int = [1 for i in range(n_row)]
            for index in range(n_discret_variable):
                result_int = result_int * (x_test[self.list_discret_variable[index]] == list_value[index])
            model_pred = self.choix_predict(x_test, result_int, model)
            result = self.add(result, model_pred)
        return result

    def choix_predict(self, x_test, result_int, model):
        """
        Pour un model, renvoie une liste dont les valeurs si result_int =0 est 0 sinon model.predict
        """
        result = []
        index_list = list(x_test.index)
        column = x_test.columns
        for index in index_list:
            if result_int[index] == 1:
                tmp = []
                for i in x_test.loc[index, :]:
                    tmp.append(i)
                data_line = pandas.DataFrame(data=[tmp], columns=column)
                result.append(model.predict(data_line)[0])
            else:
                result.append(0)
        return result

    def add(self, result, model_pred):
        for line in range(len(model_pred)):
            result[line] = result[line] + model_pred[line]
        return result


def model_regression(x_train, y_train):
    """
    linear regression
    """
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model


def model_homeType_regression(x_train, y_train):
    """
    two linear regressions according to homeType feature
    """
    model = DoubleHomeType(model_regression)
    model.fit(x_train, y_train)
    return model


def model_decision_tree(x_train, y_train):
    """
    decision tree
    """
    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)
    return model


def model_random_forest(x_train, y_train):
    """
    random forest
    """
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    return model


def model_homeType_random_forest(x_train, y_train):
    """
    two decision tree according to homeType feature
    """
    model = DoubleHomeType(model_random_forest)
    model.fit(x_train, y_train)
    return model


def model_sdfm_regression(x_train, y_train):
    """
    multiple regression according to ['input-maritalStatus', 'input-homeType']  features
    """
    list_variable_to_leaf = ['input-maritalStatus', 'input-homeType']
    model = SplitDiscreteFeatureModel(model_regression, list_variable_to_leaf)
    model.fit(x_train, y_train)
    return model


def model_sdfm_random_forest(x_train, y_train):
    """
    multiple random_forest according to ['input-maritalStatus', 'input-homeType']  features
    """
    list_variable_to_leaf = ['input-maritalStatus', 'input-homeType']
    model = SplitDiscreteFeatureModel(model_random_forest, list_variable_to_leaf)
    model.fit(x_train, y_train)
    return model
