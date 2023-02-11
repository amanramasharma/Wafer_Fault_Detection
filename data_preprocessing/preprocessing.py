import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

class Preprocessor:
    '''
    This class will be used to clean and transform the data before training.

    Written By: Aman Sharma
    Version: 1.0
    Revisions: None
    '''

    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def remove_columns(self,data,columns):
        """
            Method Name: remove_columns
            Description: This method remove the given columns from a pandas dataframe
            Output:  A pandas DataFrame after removing the specified columns.
            On Failure: Raise Exception

            Written By: Aman Sharma
            Version: 1.0
            Revisions: None
        """
        self.logger_object.log(self.file_object,"Entered the remove_columns method of the Preprocessor class")
        self.data = data
        self.columns = columns

        try:
            # drop the labels specified in the columns
            self.useful_data = self.data.drop(labels=self.columns,axis=1)
            self.logger_object.log(self.file_object,'Column removal successful.')
            self.logger_object.log(self.file_object,'Exited the remove_columns method of the Preprocessor class')
            return self.useful_data
        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured in remove_columns method of the Preprocessor class. Exception message: '+str(e))
            self.logger_object.log(self.file_object, 'Column removal Unsuccessful. Exited the remove_columns method of the Preprocessor class')
            raise e

    def separate_label_feature(self,data,label_column_name):
        """
            Method Name: separate_label_feature
            Description: This method separates the features and a Label columns.
            Output:  Returns two separate Dataframes, one containing features and the other containing Labels.
            On Failure: Raise Exception

            Written By: Aman Sharma
            Version: 1.0
            Revisions: None
        """

        self.logger_object.log(self.file_object,"Entered the separate_label_feature method of the Preprocessor class")
        try:
            # Dropped the columns specified and separate the feature columns
            self.X = data.drop(labels = label_column_name,axis = 1)
            # Filter the Label columns
            self.Y = data[label_column_name]
            self.logger_object.log(self.file_object,"Label Sepration Operation Successful. Exited the separate_label_feature method of the Preprocessor class")
            return self.X,self.Y
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occurred in separate_lable_feature method of the Preprocessor class. Exception message: '+ str(e))
            self.logger_object.log(self.file_object,
                                   "Label Sepration Operation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class")
            raise e

    def is_null_present(self,data):
        """
            Method Name: in_null_present
            Description: This method checks whether there are null values present in the pandas Dataframe or not.
            Output:  Returns a Boolean Value. True if null values are present in the DataFrame, False if they are not present.
            On Failure: Raise Exception

            Written By: Aman Sharma
            Version: 1.0
            Revisions: None
        """
        self.logger_object.log(self.file_object,"Entered the is_null_present method in Preprocessor class")
        self.null_present = False
        try:
            # Checks for the count of null values per column
            self.null_counts = data.isna().sum()
            for i in self.null_counts:
                if i > 0:
                    self.null_present = True
                    break
            if(self.null_present):
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.columns
                dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                # Storing the null coll
                dataframe_with_null.to_csv('preprocessing_data/null_values.csv')
            self.logger_object.log(self.file_object,'Finding missing values is a success. Data written to null values file. Exited the is_null_present method of the Preprocessor class')
            return self.null_present
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in is_null_present method of the Preprocessor class. Exception message: '+str(e))
            self.logger_object.log(self.file_object,'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise e

    def impute_missing_values(self, data):
        """
            Method Name: impute_missing_values
            Description: This method replaces all the missing values in the Dataframe using KNN imputer.
            Output: A Dataframe  which has all the missing values imputed.
            On Failure: Raise Exception

            Written By: Aman Sharma
            Version: 1.0
            Revisions: None
        """
        self.logger_object.log(self.file_object,'Entered the impute_missing_values method of the Preprocessor class')
        self.data = data
        try:
            imputer = KNNImputer(n_neighbors=3,weights='uniform',missing_values=np.nan)
            # Impute the missing values
            self.new_array = imputer.fit_transform(self.data)
            # Convert the nd-array returned in the step above to a Dataframe
            self.new_data = pd.DataFrame(data=self.new_array,columns=self.data.columns)
            self.logger_object.log(self.file_object,'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return self.new_data
        except Exception as e:
            self.logger_object.log(self.file_object,"Exception occurred in impute_missing_values method of the Preprocessor class. Exception message: "+str(e))
            self.logger_object.log(self.file_object,"Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class")
            raise Exception()

    def get_columns_with_zero_std_deviation(self,data):
        """
            Method Name: get_columns_with_zero_std_deviation
            Description: This method finds out the columns which have a standard deviation of zero.
            Output: List of the columns with standard deviation of zero
            On Failure: Raise Exception

            Written By: Aman Sharma
            Version: 1.0
            Revisions: None

        """
        self.logger_object.log(self.file_object,"Entered the get_columns_with_zero_std_deviation method of the Preprocessor class")
        self.columns = data.columns
        self.data_n = data.describe()
        self.col_to_drop =[]
        try:
            for x in self.columns:
                if (self.data_n[x]['std'] == 0): # Check if standard deviation is zero
                    self.col_to_drop.append(x) # prepare the list of columns with standard deviation zero
            self.logger_object.log(self.file_object,'Column search for Standard Deviation of Zero Successful. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            return self.col_to_drop

        except Exception as e:
            self.logger_object.log(self.file_object,"Exception occured in get_columns_with_zero_std_deviation method of the Preprocessor class. Exception message: "+str(e))
            self.logger_object.log(self.file_object,"Column search for Standard Deviation of Zero Failed. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class")
            raise Exception()