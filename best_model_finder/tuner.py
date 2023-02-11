from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as knn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


class Model_Finder:
    """
        This class shall be used to find the model with the best accuracy and AUC score.

        Written By: Aman Sharma
        Version: 1.0
        Revisions: None
    """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.rfc = RandomForestClassifier()
        self.xgb = XGBClassifier(objective="binary:logistic")
        self.lr = LogisticRegression()
        self.dt = DecisionTreeClassifier()
        self.svm = SVC()
        self.knn = knn()

    def get_best_params_for_random_forest(self, train_x, train_y):
        """
            Method Name: get_best_params_for_random_forest
            Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                         Use Hyper Parameter Tuning.
            Output: The Model with the best parameters
            On Failure: Raise Exception

            Written By: Aman Sharma
            Version: 1.0
            Revisions: None
        """
        self.logger_object.log(self.file_object,
                               "Entered the get_best_params_for_random_forest method of the Model_Finder class")
        try:
            # initializing with different combination of parameters
            self.param_grid = {"n_estimators": [10, 50, 100, 150, 200], "criterion": ['gini', 'entropy'],
                               "max_depth": range(2, 4, 1), "max_features": ["auto", "log2"]}

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.rfc, param_grid=self.param_grid, cv=5, verbose=3)
            # Finding the best parameters
            self.grid.fit(train_x, train_y)
            # Extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.n_estimators = self.grid.best_params_['n_estimators']

            # Creating a new model with best parameters
            self.rfc = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                              max_depth=self.max_depth, max_features=self.max_features)
            # Training the new model
            self.rfc.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   "Random Forest best params: " + str(self.grid.best_params_) +
                                   ". Exited the get_best_params_for_random_forest method of the Model_Finder class")
            return self.rfc
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   "Exception occurred in get_best_params_for_random_forest method of the "
                                   "Model_Finder class.Exception message: " + str(e))
            self.logger_object.log(self.file_object,
                                   "Random Forest Parameter tuning failed. Exited the "
                                   "get_best_params_for_random_forest method of the Model_Finder class")
            raise Exception()


    def get_best_params_for_xgboost(self, train_x, train_y):
        """
            Method Name: get_best_params_for_xgboost
            Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                             Use Hyper Parameter Tuning.
            Output: The Model with the best parameters
            On Failure: Raise Exception

            Written By: Aman Sharma
            Version: 1.0
            Revisions: None
        """
        self.logger_object.log(self.file_object,
                               "Entered the get_best_params_for_xgboost method of the Model_Finder class")
        try:
            # initializing with different combination of parameters
            self.param_grid = {
                'learning_rate': [0.5,0.1,0.01,0.001],
                'max_depth': [3, 5, 10, 20],
                "n_estimators": [10, 50, 100, 200]
            }
            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(XGBClassifier(objective="binary:logistic"),self.param_grid,verbose=3,cv=5)
            # Finding the best parameters
            self.grid.fit(train_x,train_y)
            # Extracting the best parameters
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']
            # creating a new model with the best parameters
            self.xgb = XGBClassifier(learning_rate=self.learning_rate,max_depth= self.max_depth,n_estimators=self.n_estimators)
            # training the new model
            self.xgb.fit(train_x,train_y)
            self.logger_object.log(self.file_object,
                                   "XGBoost best params: "+str(self.grid.best_params_) +
                                   ". Exited the get_best_params_for_xgboost method of the Model_Finder class.")
            return self.xgb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   "Exception occurred in get_best_params_for_xgboost method of the Model_Finder class. "
                                   "Exception message "+str(e))
            self.logger_object.log(self.file_object,
                                   "XGBoost Parameter tuning failed. Exited the get_best_params_for_xgboost method of the"
                                   " Model_Finder class")
            raise Exception()

    def get_best_params_for_LR(self,train_x,train_y):
        """
                    Method Name: get_best_params_for_LR
                    Description: get the parameters for Logistic Regression Algorithm which give the best accuracy.
                                     Use Hyper Parameter Tuning.
                    Output: The Model with the best parameters
                    On Failure: Raise Exception

                    Written By: Aman Sharma
                    Version: 1.0
                    Revisions: None
                """
        self.logger_object.log(self.file_object,"Entered the get_best_params_for_LR method of the Model_Finder class")
        try:
            # intializing with different combination of parameters
            self.param_grid = {
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'C': np.logspace(-4, 4, 20),
                'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
                'max_iter': [100, 1000, 2500, 5000]
            }
            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.lr,param_grid=self.param_grid,cv =5, verbose=3)
            # Finding the best parameters
            self.grid.fit(train_x,train_y)
            # Extracting the best parameters
            self.penalty = self.grid.best_params_['penalty']
            self.C = self.grid.best_params_['C']
            self.solver = self.grid.best_params_['solver']
            self.max_iter = self.grid.best_params_['max_iter']
            # creating a new model with best parameters
            self.log_reg = LogisticRegression(penalty=self.penalty,C=self.C,solver=self.solver,max_iter=self.max_iter)
            # training the new model
            self.log_reg.fit(train_x,train_y)
            self.logger_object.log(self.file_object,
                                   "Logistic Regression params: "+str(self.grid.best_params_)+
                                   ". Exited the get_best_params_for_LR method of the Model_Finder class.")
            return self.log_reg
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   "Exception occurred in get_best_params_for_LR method of the Model_Finder class. "
                                   "Exception message "+str(e))
            self.logger_object.log(self.file_object,
                                   "Logistic Regression Parameter tuning failed. Exited the get_best_params_for_LR"
                                   " method of the Model_Finder class")
            raise Exception()

    def get_best_params_for_DT(self,train_x,train_y):
        """
                    Method Name: get_best_params_for_DT
                    Description: get the parameters for Decision Tree Algorithm which give the best accuracy.
                                     Use Hyper Parameter Tuning.
                    Output: The Model with the best parameters
                    On Failure: Raise Exception

                    Written By: Aman Sharma
                    Version: 1.0
                    Revisions: None
                """
        self.logger_object.log(self.file_object,"Entered the get_best_params_for_DT method of the Model_Finder class")
        try:
            # initializing with different combination of parameters
            self.param_grid = {
                'max_depth': [2, 3, 5, 10, 20],
                'min_samples_leaf': [5, 10, 20, 50, 100],
                'criterion': ["gini", "entropy"]
            }
            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.dt,param_grid=self.param_grid,cv =5, verbose=3)
            # Finding the best parameters
            self.grid.fit(train_x,train_y)
            # Extracting the best parameters
            self.max_depth = self.grid.best_params_['max_depth']
            self.min_samples_leaf = self.grid.best_params_['min_samples_leaf']
            self.criterion = self.grid.best_params_['criterion']
            # creating a new model with best parameters
            self.DT = DecisionTreeClassifier(max_depth=self.max_depth,min_samples_leaf=self.min_samples_leaf,criterion=self.criterion)
            # training the new model
            self.DT.fit(train_x,train_y)
            self.logger_object.log(self.file_object,
                                   "Decision Tree params: "+str(self.grid.best_params_)+
                                   ". Exited the get_best_params_for_DT method of the Model_Finder class.")
            return self.DT
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   "Exception occurred in get_best_params_for_DT method of the Model_Finder class. "
                                   "Exception message "+str(e))
            self.logger_object.log(self.file_object,
                                   "Decision Tree Parameter tuning failed. Exited the get_best_params_for_DT"
                                   " method of the Model_Finder class")
            raise Exception()


    def get_best_params_for_SVM(self,train_x,train_y):
        """
                    Method Name: get_best_params_for_svm
                    Description: get the parameters for svm Algorithm which give the best accuracy.
                                     Use Hyper Parameter Tuning.
                    Output: The Model with the best parameters
                    On Failure: Raise Exception

                    Written By: Aman Sharma
                    Version: 1.0
                    Revisions: None
                """
        self.logger_object.log(self.file_object,"Entered the get_best_params_for_SVM method of the Model_Finder class")
        try:
            # intializing with different combination of parameters
            self.param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.svm,param_grid=self.param_grid,cv =5, verbose=3)
            # Finding the best parameters
            self.grid.fit(train_x,train_y)
            # Extracting the best parameters
            self.C = self.grid.best_params_['C']
            self.gamma = self.grid.best_params_['gamma']
            self.kernel = self.grid.best_params_['kernel']
            # creating a new model with best parameters
            self.svm = SVC(C=self.C,gamma=self.gamma,kernel=self.kernel)
            # training the new model
            self.svm.fit(train_x,train_y)
            self.logger_object.log(self.file_object,
                                   "SVM params: "+str(self.grid.best_params_)+
                                   ". Exited the get_best_params_for_SVM method of the Model_Finder class.")
            return self.svm
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   "Exception occurred in get_best_params_for_SVM method of the Model_Finder class. "
                                   "Exception message "+str(e))
            self.logger_object.log(self.file_object,
                                   "SVM Parameter tuning failed. Exited the get_best_params_for_SVM"
                                   " method of the Model_Finder class")
            raise Exception()

    def get_best_params_for_knn(self,train_x,train_y):
        """
                    Method Name: get_best_params_for_knn
                    Description: get the parameters for K-Nearest Neighbour Algorithm which give the best accuracy.
                                     Use Hyper Parameter Tuning.
                    Output: The Model with the best parameters
                    On Failure: Raise Exception

                    Written By: Aman Sharma
                    Version: 1.0
                    Revisions: None
                """
        self.logger_object.log(self.file_object,"Entered the get_best_params_for_knn method of the Model_Finder class")
        try:
            # intializing with different combination of parameters
            self.param_grid = {'leaf_size': list(range(1, 50)) , 'n_neighbors' : list(range(1, 30)), 'p' : [1, 2],
                               'weights' : ['uniform','distance'] ,'metric' : ['minkowski','euclidean','manhattan']}
            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.knn,param_grid=self.param_grid,cv =5, verbose=3)
            # Finding the best parameters
            self.grid.fit(train_x,train_y)
            # Extracting the best parameters
            self.leaf_size = self.grid.best_params_['leaf_size']
            self.n_neighbors = self.grid.best_params_['n_neighbors']
            self.p = self.grid.best_params_['p']
            self.weights = self.grid.best_params_['weights']
            self.metric = self.grid.best_params_['metric']
            # creating a new model with best parameters
            self.KNN = knn(leaf_size=self.leaf_size,n_neighbors=self.n_neighbors,p=self.p,weights=self.weights,metric=self.metric)
            # training the new model
            self.KNN.fit(train_x,train_y)
            self.logger_object.log(self.file_object,
                                   "KNN params: "+str(self.grid.best_params_)+
                                   ". Exited the get_best_params_for_knn method of the Model_Finder class.")
            return self.KNN
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   "Exception occurred in get_best_params_for_knn method of the Model_Finder class. "
                                   "Exception message "+str(e))
            self.logger_object.log(self.file_object,
                                   "KNN Parameter tuning failed. Exited the get_best_params_for_knn"
                                   " method of the Model_Finder class")
            raise Exception()


    def get_best_model(self,train_x,train_y,test_x,test_y):

        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')

        try:
            # create best model for XGBoost
            self.xgboost = self.get_best_params_for_xgboost(train_x,train_y)
            self.prediction_xgboost = self.xgboost.predict(test_x) # predictions using the XGBoost Model

            if len(test_y.unique()) == 1: # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.xgboost_score = accuracy_score(test_y,self.prediction_xgboost)
                self.logger_object.log(self.file_object,"Accuracy for XGBoost: "+str(self.xgboost_score)) # Log AUC
            else:
                self.xgboost_score = roc_auc_score(test_y, self.prediction_xgboost)
                self.logger_object.log(self.file_object,"AUC for XGBoost: " +str(self.xgboost_score))

            # create best model for Random forest
            self.rf = self.get_best_params_for_random_forest(train_x,train_y)
            self.prediction_rf  = self.rf.predict(test_x) # predictions using the Random Forest Model

            if len(test_y.unique()) ==1: # if there is only one label in y , then  roc_auc_score returns error. We will use accuracy in that case
                self.rf_score = accuracy_score(test_y, self.prediction_rf)
                self.logger_object.log(self.file_object, "Accuracy for Random Forest: " + str(self.rf_score))  # Log AUC
            else:
                self.rf_score = roc_auc_score(test_y, self.prediction_rf)
                self.logger_object.log(self.file_object, "AUC for Random Forest: " +str(self.rf_score))

            # create best model for Logistic regression
            self.lr = self.get_best_params_for_LR(train_x, train_y)
            self.prediction_lr = self.lr.predict(test_x)  # predictions using the Logistic Regression

            if len(test_y.unique()) == 1:  # if there is only one label in y , then  roc_auc_score returns error. We will use accuracy in that case
                self.lr_score = accuracy_score(test_y, self.prediction_lr)
                self.logger_object.log(self.file_object, "Accuracy for Logistic Regression: " + str(self.lr_score))  # Log AUC
            else:
                self.lr_score = roc_auc_score(test_y, self.prediction_lr)
                self.logger_object.log(self.file_object, "AUC for Logistic Regression: " +str(self.lr_score))

            # create best model for Decision Tree
            self.dt = self.get_best_params_for_DT(train_x, train_y)
            self.prediction_dt = self.dt.predict(test_x)  # predictions using the Decision Tree

            if len(test_y.unique()) == 1:  # if there is only one label in y , then  roc_auc_score returns error. We will use accuracy in that case
                self.dt_score = accuracy_score(test_y, self.prediction_dt)
                self.logger_object.log(self.file_object, "Accuracy for Decision Tree: " + str(self.dt_score))  # Log AUC
            else:
                self.dt_score = roc_auc_score(test_y, self.prediction_dt)
                self.logger_object.log(self.file_object, "AUC for Decision Tree: " +str(self.dt_score))

            # create best model for SVM
            self.svm = self.get_best_params_for_SVM(train_x, train_y)
            self.prediction_svm = self.svm.predict(test_x)  # predictions using the svm

            if len(test_y.unique()) == 1:  # if there is only one label in y , then  roc_auc_score returns error. We will use accuracy in that case
                self.svm_score = accuracy_score(test_y, self.prediction_svm)
                self.logger_object.log(self.file_object, "Accuracy for SVM: " + str(self.svm_score))  # Log AUC
            else:
                self.svm_score = roc_auc_score(test_y, self.prediction_svm)
                self.logger_object.log(self.file_object, "AUC for SVM: " +str(self.svm_score))

            # create best model for KNN
            self.knn = self.get_best_params_for_knn(train_x, train_y)
            self.prediction_knn = self.knn.predict(test_x)  # predictions using the knn

            if len(test_y.unique()) == 1:  # if there is only one label in y , then  roc_auc_score returns error. We will use accuracy in that case
                self.knn_score = accuracy_score(test_y, self.prediction_knn)
                self.logger_object.log(self.file_object, "Accuracy for knn: " + str(self.knn_score))  # Log AUC
            else:
                self.knn_score = roc_auc_score(test_y, self.prediction_knn)
                self.logger_object.log(self.file_object, "AUC for knn: " +str(self.svm_score))

            # Comparing the all the models
            model_score = [self.xgboost_score,self.rf_score,self.lr_score,self.dt_score,self.svm_score,self.knn_score]
            best_model = max(model_score)
            if best_model == self.xgboost_score:
                return "XGBoost ",self.xgboost
            elif best_model == self.rf_score:
                return "Random Forest ",self.rf
            elif best_model == self.lr_score:
                return "Logistic Regression ", self.lr
            elif best_model == self.dt_score:
                return "Decision Tree ", self.dt
            elif best_model == self.svm_score:
                return  "SVM ",self.svm

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
        raise Exception()

 