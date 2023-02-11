import pandas as pd

class Data_Getter_Pred:
    """
        This class shall be used for obtaining the data from the source for prediction.

        Written By: Aman Sharma
        Version: 1.0
        Revisions: None
    """
    def __init__(self, file_object, logger_object):
        self.prediction_file='Prediction_FileFromDB/InputFile.csv'
        self.file_object=file_object
        self.logger_object=logger_object
    def get_data(self):
        """
        Method Name: get_data
        Description: This method reads data from source.
        Output: A pandas DataFrame 
        On Failure: Raise Exception

        Written By: Aman Sharma
        Version: 1.0
        Revisions: None
        """
        self.logger_object.log(self.file_object,"Entered the get_data method of the Data_Getter class")
        try:
            self.data = pd.read_csv(self.prediction_file) # reading the data file
            self.logger_object.log(self.file_object,"Data Load Successfully. Exited the get_data method of the Data_Getter class")
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,"Exception occured in get_data method of the Data_Getter class. Exception message: "+str(e))
            self.logger_object.log(self.file_object,"Data load unsuccessful. Exited the get_data method of the Data_Getter class")
            raise Exception