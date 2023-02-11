from datetime import datetime
from Training_Raw_data_validation.rawValidation import Raw_Data_validation
from DataTypeValidation_Insertion_Training.DataTypeValidation import dBOperation
from DataTransform_Training.DataTransformation import dataTransform
from application_logging import logger

class train_validation:
    def __init__(self,path):
        self.raw_data = Raw_Data_validation(path)
        self.dataTransform = dataTransform()
        self.dBOperation = dBOperation()
        self.file_object = open("Training_Logs/Training_Main_Log.txt",'a+')
        self.log_writer = logger.App_Logger()

    def train_validation(self):
        try:
            # Validation
            # Log
            self.log_writer.log(self.file_object,'Start of Validation on files!!')
            # Extracting values from prediction schema
            LengthOfDataStampInFile, LengthOfTimeStampInFile, column_names,noofcolumns = self.raw_data.valuesFromSchema()
            # Getting the regex defined to validate filename
            regex = self.raw_data.manualRegexCreation()
            # Validating filename of prediction files
            self.raw_data.validationFileNameRaw(regex,LengthOfDataStampInFile,LengthOfTimeStampInFile)
            # Validating column length in the file
            self.raw_data.validateColumnLength(noofcolumns)
            # Validating if any column has all values missing
            self.raw_data.validateMissingValuesInWholeColumn()
            # Log
            self.log_writer.log(self.file_object, 'Raw Data Validation Complete!!')

            # Data Transfromation
            # Log
            self.log_writer.log(self.file_object, 'Starting Data Transformation!!!')
            # Replacing blanks values with "Null" values to insert in table
            self.dataTransform.replaceMissingWithNull()
            # Log
            self.log_writer.log(self.file_object, 'Data Transformation Completed!!!')

            # DataBase Operation
            # Log
            self.log_writer.log(self.file_object, 'Creating Tarining_Database and tables on the basis of given schema!!')
            # Create database with given name, if present open the connection! Create table with columns according to schema
            self.dBOperation.createTableDB('Training',column_names)
            # Log
            self.log_writer.log(self.file_object, 'Table Creation Completed!!!')
            self.log_writer.log(self.file_object, 'Insertion of Data into Table started!!!')
            # Insert csv files in table
            self.dBOperation.insertIntoTableGoodData('Training')
            # Log
            self.log_writer.log(self.file_object, 'Insertion in Table completed!!!')

            # Delete Good and Bad Data Folder
            # Log
            self.log_writer.log(self.file_object, 'Deleting Good Data Folder!!!')
            # Delete the good data folder after loading files in table
            self.raw_data.deleteExistingGoodDataTrainingFolder()
            # Log
            self.log_writer.log(self.file_object, 'Good_data folder deleted!!!')
            self.log_writer.log(self.file_object, 'Deleting Bad_data folder after moving the files into Archive!!!')
            # Move the bad files to archive folder
            self.raw_data.moveBadFilesToArchiveBad()
            # Log
            self.log_writer.log(self.file_object, 'Bad Files moved to Archive!!! and Bad folder Deleted!!!')
            self.log_writer.log(self.file_object, 'Validation Operation completed!!!')

            # Extract files from table
            # Log
            self.log_writer.log(self.file_object, 'Extracting csv file from table!!!')
            # Export data in table to csvfile
            self.dBOperation.selectingDatafromtableintocsv('Training')
            self.file_object.close()

        except Exception as e:
            raise e
