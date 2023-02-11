from datetime import datetime
from os import listdir
import pandas as pd
from application_logging.logger import App_Logger

class dataTransform:
    """
        This class will be used for transforming the Good Raw Training Data before loading it in Database!!.

        Written By: Aman Sharma
        Version: 1.0
        Revisions: None
    """
    def __init__(self):
        self.goodDataPath = "Training_Raw_files_validated/Good_Raw"
        self.logger = App_Logger()

    def replaceMissingWithNull(self):
        """
            Method Name: replaceMissingWithNull
            Description: This method replace missing values with "NULL" to store in the table.
                         Using Substring in the first column to keep only "Integer" data

            Written by: Aman Sharma
            Version: 1.0
            Revisions: None
        """
        log_file = open("Training_Logs/dataTransformLog.txt", 'a+')
        try:
            onlyfiles = [f for f in listdir(self.goodDataPath)]
            for file in onlyfiles:
                csv = pd.read_csv(self.goodDataPath+"/"+file)
                csv.fillna("NULL", inplace=True)
                csv['Wafer'] = csv['Wafer'].str[6:]
                csv.to_csv(self.goodDataPath+"/"+file, index = None, header= True)
                self.logger.log(log_file, " %s : File Transformed Successfully!!" % file)
        except Exception as e:
            self.logger.log(log_file, "Data Transformation failed because:: %s " % e)
            log_file.close()
        log_file.close()
