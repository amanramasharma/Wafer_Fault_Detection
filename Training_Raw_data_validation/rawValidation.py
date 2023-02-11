import sqlite3
from datetime import datetime
from os import listdir
import os
import re
import json
import shutil
import pandas as pd
from application_logging.logger import App_Logger


class Raw_Data_validation:
    """
            This class shall be used  for handling all the validation done on the Raw Training Data.

            Written By: Aman Sharma
            Version: 1.0
            Revisions: None
    """

    def __init__(self, path):
        self.Batch_Directory = path
        self.schema_path = "schema_training.json"
        self.logger = App_Logger()

    def valuesFromSchema(self):
        """
                    Method Name: valuesFromSchema
                    Description: This method extracts all the relevant information from the pre-defined "Schema" file.
                    Output: LengthOfDateStampInFile, LengthOfTimeStampFile, column_names, Number of Columns
                    On Failure: Raise ValueError, KeyError, Exception

                    Written By: Aman Sharma
                    Version: 1.0
                    Revisions: None

        """

        try:
            with open(self.schema_path, 'r') as f:
                dic = json.load(f)
                f.close()
            pattern = dic['SampleFileName']
            LengthOfDateStampInFile = dic['LengthOfDateStampInFile']
            LengthOfTimeStampInFile = dic['LengthOfTimeStampInFile']
            column_names = dic['ColName']
            NumberofColumns = dic['NumberofColumns']

            file = open("Training_Logs/valuesfromSchemaValidationLog.txt", 'a+')
            message = "LengthOfDateStampInFile:: %s" % LengthOfDateStampInFile + '\t' + "LengthOfTimeStampInFile :: %s " % LengthOfTimeStampInFile + "\t" + "NumberofColumns:: %s" % NumberofColumns + "\n"
            self.logger.log(file, message)
            file.close()

        except ValueError:
            file = open("Training_Logs/valuesfromSchemaValidationLog.txt", 'a+')
            self.logger.log(file, "ValueError: Value not found in schema_training.json")
            file.close()
            raise ValueError

        except KeyError:
            file = open("Training_Logs/valuesfromSchemaValidationLog.txt", 'a+')
            self.logger.log(file, "KeyError: key value error incorrect key passed")
            file.close()
            raise KeyError

        except Exception as e:
            file = open("Training_Logs/valuesfromSchemaValidationLog.txt", 'a+')
            self.logger.log(file, str(e))
            file.close()
            raise e

        return LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, NumberofColumns

    def manualRegexCreation(self):
        """
            Method Name: manualRegexCreation
            Description: This method contains manually defined regex based on the "FileName" given in "Schema" file.
                         This is used to validate the filename of the tarinng data.
            Output: Regex pattern
            On Failure: None

            Written By: Aman Sharma
            Versions: 1
            Revisions: None

        """
        regex = "['wafer']+['\_'']+[\d_]+[\d]+\.csv"
        return regex

    def createDirectoryForGoodBadRawData(self):

        """
            Method Name: createDirectoryForGoodBadRawData
            Description: This method creates directories to store the Good Data and Bad Data
                         after validation the training data.
            Output: None
            On Failure: OSError

            Written By: Aman Sharma
            Version: 1.0
            Revisions: None
        """
        try:
            path = os.path.join("Training_Raw_files_validated/", "Good_Raw/")
            if not os.path.isdir(path):
                os.makedirs(path)
            path = os.path.join("Training_Raw_files_validated/", "Bad_Raw/")
            if not os.path.isdir(path):
                os.makedirs(path)
        except OSError as ex:
            file = open("Training_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, "Error while creating Directory: %s" % ex)
            file.close()
            raise OSError

    def deleteExistingGoodDataTrainingFolder(self):
        """
            Method Name: deleteExistingGoodDataTrainingFolder
            Description: This Method is used to delete the Good Raw Directory after store the Good Data into the table.
                         Once the good files are loaded in the DB, deleting the directory ensures the space optimization
            Output: None
            On Failure: Raise OS Error

            Written By: Aman Sharma
            Versions: 1.0
            Revisions : None
        """

        try:
            path = 'Training_Raw_files_validated/'
            if os.path.isdir(path + 'Good_Raw/'):
                shutil.rmtree(path + 'Good_Raw/')
                file = open("Training_Logs/GeneralLog.txt", 'a+')
                self.logger.log(file, "Good Raw Directory is deleted successfully !!!")
                file.close()
        except OSError as s:
            file = open("Training_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, "Error while Deleting Directory : %s " % s)
            file.close()
            raise OSError

    def deleteExistingBadDataTrainingFolder(self):
        """
            Method Name: deleteExistingBadDataTrainingFolder
            Description: This Method is used to delete the Bad Raw Directory.
            Output: None
            On Failure: Raise OS Error

            Written By: Aman Sharma
            Verisons: 1.0
            Revisions : None
        """

        try:
            path = 'Training_Raw_files_validated/'
            if os.path.isdir(path + 'Bad_Raw/'):
                shutil.rmtree(path + 'Bad_Raw/')
                file = open("Training_Logs/GeneralLog.txt", 'a+')
                self.logger.log(file, "Bad Raw Directory is deleted successfully !!!")
                file.close()
        except OSError as s:
            file = open("Training_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, "Error while Deleting Directory : %s " % s)
            file.close()
            raise OSError

    def moveBadFilesToArchiveBad(self):
        """
            Method Name: moveBadFilesToArchiveBad
            Description: This method move Bad Raw files into Bad Archive then delete Bad Raw directory.
                         Archive the bad files to send them back to client for invalid data issue.
            Output: None
            On Failure: OSError

            Written By: Aman Sharma
            Versions: 1.0
            Revisions:  None
        """

        now = datetime.now()
        date = now.date()
        time = now.strftime("%H%M%S")
        try:
            source = 'Training_Raw_files_validated/Bad_Raw/'
            if os.path.isdir(source):
                path = "TrainingArchiveBadData"
                if not os.path.isdir(path):
                    os.makedirs(path)
                dest = "TrainingArchiveBadData/BadData_" + str(date) + "_" + str(time)
                if not os.path.isdir(dest):
                    os.makedirs(dest)
                files = os.listdir(source)
                for f in files:
                    if f not in os.listdir(dest):
                        shutil.move(source + f, dest)
                file = open("Training_Logs/GeneralLog.txt", 'a+')
                self.logger.log(file, "Bad files moved to archive")
                path = "Training_Raw_files_validated"
                if os.path.isdir(path + 'Bad_Raw/'):
                    shutil.rmtree(path + "Bad_Raw/")
                self.logger.log(file, "Bad Raw Folder Deleted Successfully!!")
                file.close()
        except Exception as e:
            file = open("Training_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, "Error while moving bad files to archive:: %s" % e)
            file.close()
            raise e

    def validationFileNameRaw(self, regex, LengthOfDateStampInFile, LengthOfTimeStampFile):
        """
            Method Name: validationFileNameRaw
            Description: This Method is used to validate the file name of training csv files as per given name in the schema!
                         Regex pattern is used for validation. If name format is not match the file name is moved to Bad Raw Data
                         Folder else in Good Raw Data Folder.
            Output: None
            On Failure: Exception

            Written By: Aman Sharma
            Verison: 1.0
            Revsisions: None
        """

        # pattern "['wafer']+['\_'']+[\d_]+[\d]+\.csv"
        # delete the directories for good and bad data in case last run was unsuccesful and folders were not deleted.
        self.deleteExistingBadDataTrainingFolder()
        self.deleteExistingGoodDataTrainingFolder()
        # Create new directories
        self.createDirectoryForGoodBadRawData()
        onlyfiles = [f for f in listdir(self.Batch_Directory)]
        try:
            f = open("Training_Logs/nameValidationLog.txt", 'a+')
            for filename in onlyfiles:
                if (re.match(regex, filename)):
                    splitAtDot = re.split('.csv', filename)
                    splitAtDot = (re.split('_', splitAtDot[0]))
                    if len(splitAtDot[1]) == LengthOfDateStampInFile:
                        if len(splitAtDot[2]) == LengthOfTimeStampFile:
                            shutil.copy("Training_Batch_Files/" + filename, "Training_Raw_files_validated/Good_Raw")
                            self.logger.log(f, "Valid file name!! File moved to Good Raw Folder :: %s " % filename)
                        else:
                            shutil.copy("Training_Batch_Files/" + filename, "Training_Raw_files_validated/Bad_Raw")
                            self.logger.log(f, "Invalid file name!! File moved to Bad Raw Folder :: %s " % filename)
                    else:
                        shutil.copy("Training_Batch_Files/" + filename, "Training_Raw_files_validated/Bad_Raw")
                        self.logger.log(f, "Invalid file name!! File moved to Bad Raw Folder :: %s " % filename)
                else:
                    shutil.copy("Training_Batch_Files/" + filename, "Training_Raw_files_validated/Bad_Raw")
                    self.logger.log(f, "Invalid file name!! File moved to Bad Raw Folder :: %s " % filename)
            f.close()

        except Exception as e:
            f = open("Training_Logs/nameValidationLog.txt", 'a+')
            self.logger.log(f, "Error Occurred while validating FileName %s" % e)
            f.close()
            raise e

    def validateColumnLength(self, NumberofColumns):
        """
            Method Name: validateColumnLength
            Description: This method validates the Number of columns in the csv files.
                         It is should be same as given in the schema file.
                         If number of columns is not same as mentioned in schema file then it'll move to Bad Raw Folder
                         else kept in Good data.
            Output: None
            On Failure: Exception

            Written By: Aman Sharma
            Verison: 1.0
            Revsisions: None

        """
        try:
            f = open("Training_Logs/columnValidationLog.txt", 'a+')
            self.logger.log(f,"Column Length Validation Started!!")
            for file in listdir('Training_Raw_files_validated/Good_Raw/'):
                csv = pd.read_csv("Training_Raw_files_validated/Good_Raw/" + file)
                if csv.shape[1] == NumberofColumns:
                    pass
                else:
                    shutil.move("Training_Raw_files_validated/Good_Raw/" + file, "Training_Raw_files_validated/Bad_Raw")
                    self.logger.log(f, "Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)
            self.logger.log(f, "Column Length Validation Completed!!")
        except OSError:
            f = open("Training_Logs/columnValidationLog.txt", 'a+')
            self.logger.log(f, "Error Occured while moving the file :: %s" % OSError)
            f.close()
            raise OSError
        except Exception as e:
            f = open("Training_Logs/columnValidationLog.txt", 'a+')
            self.logger.log(f, "Error Occured:: %s" % e)
            f.close()
            raise e
        f.close()

    def validateMissingValuesInWholeColumn(self):
        """

            Method Name: validateMissingValuesInWholeColumn
            Description: This Method validates if any column in the csv file has all values missing.
                         If all the values are missing, the file is not suitable for processing.
                         Such Files are moved to Bad Raw Data
            Output: None
            On Failure: Exception

            Written By: Aman Sharma
            Verison: 1.0
            Revsisions: None

        """
        try:
            f = open("Training_Logs/missingValuesInColumn.txt", "a+")
            self.logger.log(f, "Missing Values Validation Started!!")

            for file in listdir("Training_Raw_files_validated/Good_Raw/"):
                csv = pd.read_csv("Training_Raw_files_validated/Good_Raw/" + file)
                count = 0
                for columns in csv:
                    if (len(csv[columns]) - csv[columns].count()) == len(csv[columns]):
                        count += 1
                        shutil.move("Training_Raw_files_validated/Good_Raw/" + file,
                                    "Training_Raw_files_validated/Bad_Raw")
                        self.logger.log(f, "Invalid Column Length for the file!! File Moved to Bad Raw Folder :: %s" % file)
                        break
                if count == 0:
                    csv.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)
                    csv.to_csv("Training_Raw_files_validated/Good_Raw/" + file, index=None, header=True)
        except OSError:
            f = open("Training_Logs/missingValuesInColumn.txt", "a+")
            self.logger.log(f, "Error Occurred while moving the file :: %s" % OSError)
            f.close()
            raise OSError
        except Exception as e:
            f = open("Training_Logs/missingValuesInColumn.txt", "a+")
            self.logger.log(f, "Error Occurred %s" % e)
            f.close()
            raise e
        f.close()
