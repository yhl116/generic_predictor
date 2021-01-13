import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

class DataPreprocessing:
    def __init__(self, train_data_file, test_data_file):
        self.__raw_train_data = pd.read_csv(train_data_file, index_col = 0, header = 1)
        self.__raw_test_data = pd.read_csv(test_data_file, index_col = 0, header = 1)
        self.__y_train = self.__raw_train_data[self.__raw_train_data.columns[-1]].tolist()
        self.__y_test = self.__raw_test_data[self.__raw_test_data.columns[-1]].tolist()
        self.__scaled_x_train, self.__scaled_x_test = self.__scaler()
        
    def __scaler(self):
        scaler = StandardScaler()
        scaled_x_train = scaler.fit_transform(self.__raw_train_data[self.__raw_train_data.columns[:-1]])
        scaled_x_test = scaler.transform(self.__raw_test_data[self.__raw_test_data.columns[:-1]])
        return scaled_x_train, scaled_x_test
    
    def get_x_train(self):
        return self.__scaled_x_train
    
    def get_x_test(self):
        return self.__scaled_x_test
    
    def get_y_test(self):
        return self.__y_test
    
    def get_y_train(self):
        return self.__y_train
    
    def get_scaled_data(self):
        return self.__scaled_x_train, self.__y_train, self.__scaled_x_test, self.__y_test

class GenericClassifier:
    '''
    Attributes:
    __x_train : n-Dimensional float array
        IV for train data
    __y_train : 1D float array
        DV for test data
    __x_test : n-Dimensional float array
        IV for test data
    __y_test : 1D float array
        DV for test data
    __kfold_index : 1D integer array
        [train_index, test_index]
    __results : {string : dict}
        key : model name
        value : model result dict
            "Acc" : DataFrame - Training and testing accuracy for 5 kfolds
            "Confusion Matrix" : 3D float array - Confusion matrix for 5 kfolds
            "Classification Report" : Classification Report dict list - Classification report for 5 kfolds
            "Mean Training Acc" : float
            "Mean Testing Acc" : float
            "Mean Confusion Matrix" : 2D foat array
    '''
    def __init__(self, x_train, y_train, x_test, y_test):
        self.__x_train = x_train
        self.__y_train = np.asarray(y_train)
        self.__x_test = x_test
        self.__y_test = np.asarray(y_test)
        self.__data = x_train, y_train, x_test, y_test
        self.__results = dict()
        self.__existing_models = set()

        # initialise kfold split
        kf = KFold(n_splits = 5, shuffle = True, random_state = 2)
        self.__kfold_index = list(kf.split(x_train)) 

        # model_list
        self.model_list = {
            "Logistic Regression": LogisticRegression(),
            "Gradient Boosting Classifier": GradientBoostingClassifier(),
            "Random Forest Classifier": RandomForestClassifier(),
            "Support Vector Classifier": SVC()
        }

    def get_results(self):
        return self.__results

    def kfold_modeling(self, input_model, input_model_name, x, y):
        '''
        Higher level function which performs kfold validation on `modeling`

        Parameters:
        input_model : model object
            - e.g. SVC(), RandomForest()

        input_model_name : str

        x : 2D array

        y : 2D array

        Side effects:
        Update self.__results[input_model_name] with validation results.
        '''

        def modeling(input_model, data):
            '''Performs modeling on given data'''
            results = dict()
            input_model.fit(data[0], data[1])
            results["Training Acc"] = input_model.score(data[0], data[1])
            results["Testing Acc"] = input_model.score(data[2], data[3])
            
            pred = input_model.predict(data[2])
            results["Confusion Matrix"] = confusion_matrix(data[3], pred)
            results["Classification Report"] = classification_report(data[3], pred)

            return results

        temp_results = [modeling(input_model, (x[train_index], y[train_index], x[test_index], y[test_index])) for train_index, test_index in self.__kfold_index]

        acc_dict = {
            "Training Acc": [fold_result["Training Acc"] for fold_result in temp_results],
            "Testing Acc": [fold_result["Testing Acc"] for fold_result in temp_results]
        }

        final_results = dict()
        final_results["Acc"] = pd.DataFrame(acc_dict)
        final_results["Confusion Matrix"] = [fold_result["Confusion Matrix"] for fold_result in temp_results]
        final_results["Classification Report"] = [fold_result["Classification Report"] for fold_result in temp_results]

        # Mean of results
        mean_acc_df = final_results["Acc"].mean()
        final_results["Mean Training Acc"] = mean_acc_df["Training Acc"]
        final_results["Mean Testing Acc"] = mean_acc_df["Testing Acc"]
        final_results["Mean Confusion Matrix"] = np.mean(final_results["Confusion Matrix"], axis = 0)
        # final_results["Mean Classification Report"] = np.mean(final_results["Classification Report"], axis = 0)

        # Add processed results here

        # Add processed results here end

        self.__results[input_model_name] = final_results
        self.__existing_models.add(input_model_name)

    def run_models(self, run_all = False):
        '''
        High level function to train models

        parameters:
        - run_all : bool
            True: 
                - Run all the models in self.model_list
                - Reuse kfold split index
                - Will retrain existing models
            False: 
                - Only run new models added into self.model_list
                - Create new kfold split index
                - Retain and reuse results of existing models
        '''

        if run_all:
            kf = KFold(n_splits = 5, shuffle = True, random_state = 2)
            self.__kfold_index = list(kf.split(x_train)) 
            models_to_run = self.model_list
        else:
            # only take models which have not been run
            models_to_run = dict()
            for model_name, model in self.model_list.items():
                if model_name not in self.__existing_models:
                    models_to_run[model_name] = model

        for model_name, model in models_to_run.items():
            self.kfold_modeling(model, model_name, self.__x_train, self.__y_train)

    def compare_models(self):
        '''
        Compare diferent models

        returns : DataFrame


        '''
        compared_attributes = [
            "Mean Training Acc",
            "Mean Testing Acc"
        ]
        
        compare_models_dict = {model: [model_results["Mean Training Acc"], model_results["Mean Testing Acc"]] for model, model_results in self.__results.items()}
        compare_models_df = pd.DataFrame(compare_models_dict, index = compared_attributes)
        
        return compare_models_df.transpose()

    def save_models(self, file_name):
        '''
        Save your work progress

        To restore your object state use the following coded:
        with open(file_name, "r") as f:
            obj = pickle.load(f)[1]
        '''

        file_name = file_name + ".pickle"
        f = open(file_name, "w")
        pickle.dump(self,f)
        f.close
        
    def get_x_train(self):
        return self.__x_train
    
    def get_x_test(self):
        return self.__x_test
    
    def get_y_test(self):
        return self.__y_test
    
    def get_y_train(self):
        return self.__y_train
    
    def get_data(self):
        return self.__data