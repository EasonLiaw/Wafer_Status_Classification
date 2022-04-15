import os, shutil, json
import pandas as pd
import mysql.connector
import csv
from Application_Logger.logger import App_Logger
import DBConnectionSetup as login

class DBOperations:
    def __init__(self, tablename, file_object):
        self.tablename = tablename
        self.file_object = file_object
        self.log_writer = App_Logger()
        self.host = login.logins['host']
        self.user = login.logins['user']
        self.password = login.logins['password']
        self.dbname = login.logins['dbname']

    def newDB(self, schema):
        self.log_writer.log(self.file_object, f"Start creating new table({self.tablename}) in SQL database ({self.dbname})")
        self.schema = schema
        try:
            conn = mysql.connector.connect(host=self.host,user=self.user,password=self.password,database=self.dbname)
            mycursor = conn.cursor()
            for name, type in zip(self.schema['ColName'].keys(),self.schema['ColName'].values()):
                mycursor.execute(f"""SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{self.tablename}'""")
                if mycursor.fetchone()[0] == 1:
                    try:
                        mycursor.execute(f"ALTER TABLE {self.tablename} ADD {name} {type}")
                        self.log_writer.log(self.file_object, f"Column {name} added into {self.tablename} table")
                    except:
                        self.log_writer.log(self.file_object, f"Column {name} already exists in {self.tablename} table")
                else:
                    try:
                        mycursor.execute(f"CREATE TABLE {self.tablename} ({name} {type})")
                        self.log_writer.log(self.file_object, f"{self.tablename} table created with column {name}")
                    except Exception as e:
                        self.log_writer.log(self.file_object, f"SQL server has error of creating new table {self.tablename} with the following error: {e}")
                        raise Exception()
        except ConnectionError:
            self.log_writer.log(self.file_object, "Error connecting to SQL database")
            raise Exception()
        except Exception as e:
            self.log_writer.log(self.file_object, f"The following error occured when connecting to SQL database: {e}")
            raise Exception()
        conn.close()
        self.log_writer.log(self.file_object, f"Finish creating new table({self.tablename}) in SQL database ({self.dbname})")
    
    def data_insert(self, gooddir):
        self.log_writer.log(self.file_object, "Start inserting new good training data into SQL database")
        self.gooddir = gooddir
        try:
            conn = mysql.connector.connect(host=self.host,user=self.user,password=self.password,database = self.dbname)
            mycursor = conn.cursor()
            for file in os.listdir(self.gooddir[:-1]):
                with open(self.gooddir+file, "r") as f:
                    next(f)
                    filename = csv.reader(f)
                    for line in enumerate(filename):
                        try:
                            line[1][0] = f"\"{line[1][0]}\""
                            mycursor.execute(f"INSERT INTO {self.tablename} VALUES ({','.join(line[1])})")
                            conn.commit()
                        except Exception as e:
                            self.log_writer.log(self.file_object, f'Row {line[0]} could not be inserted into database for {file} file')
                            conn.rollback()
                            raise Exception()
                    self.log_writer.log(self.file_object, f"{file} file added into database")
        except ConnectionError:
            self.log_writer.log(self.file_object, "Error connecting to SQL database")
            raise Exception()
        except Exception as e:
            self.log_writer.log(self.file_object, f"The following error occured when connecting to SQL database: {e}")
            raise Exception()
        conn.close()
        self.log_writer.log(self.file_object, "Finish inserting new good training data into SQL database")

    def compile_data_from_DB(self,compiledir):
        self.log_writer.log(self.file_object, "Start writing compiled good training data into a new CSV file")
        self.compiledir = compiledir
        try:
            conn = mysql.connector.connect(host=self.host,user=self.user,password=self.password,database = self.dbname)
            try:
                data = pd.read_sql(f'''SELECT DISTINCT * FROM {self.tablename};''', conn)
                data.to_csv(self.compiledir, index=False)
            except Exception as e:
                self.log_writer.log(self.file_object, f"File exporting failed with the following error: {e}")
                raise Exception()
        except ConnectionError:
            self.log_writer.log(self.file_object, "Error connecting to SQL database")
            raise Exception()
        except Exception as e:
            self.log_writer.log(self.file_object, f"The following error occured when connecting to SQL database: {e}")
            raise Exception()
        conn.close()
        self.log_writer.log(self.file_object, "Finish writing compiled good training data into a new CSV file")

class rawtraindatavalidation(DBOperations):
    def __init__(self, tablename, file_object, gooddir, baddir):
        super().__init__(tablename, file_object)
        self.gooddir = gooddir
        self.baddir = baddir
        self.log_writer = App_Logger()

    def load_train_schema(self, filename):
        self.log_writer.log(self.file_object, "Start loading train schema")
        self.filename = filename
        try:
            with open(filename, 'r') as f:
                schema = json.load(f)
        except Exception as e:
            self.log_writer.log(self.file_object, f"Training schema fail to load with the following error: {e}")
            raise Exception()
        self.log_writer.log(self.file_object, "Finish loading train schema")
        return schema
    
    def file_initialize(self, filelist):
        self.log_writer.log(self.file_object, "Start initializing folder structure")
        self.filelist = filelist
        for folder in self.filelist:
            try:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                os.makedirs(os.path.dirname(folder), exist_ok=True)
                self.log_writer.log(self.file_object, f"Folder {folder} has been initialized")
            except Exception as e:
                self.log_writer.log(self.file_object, f"Folder {folder} could not be initialized with the following error: {e}")
                raise Exception()
        self.log_writer.log(self.file_object, "Finish initializing folder structure")
    
    def file_namecheck(self,sourcedir,schema):
        self.log_writer.log(self.file_object, "Start checking for valid name of files")
        self.sourcedir = sourcedir
        self.schema = schema
        for file in os.listdir(self.sourcedir):
            filename = file.split(".csv")[0].split('_')
            if len(filename)!=3 or filename[0] != 'wafer' or len(filename[1])!=self.schema['LengthOfDateStampInFile'] or len(filename[2])!=self.schema['LengthOfTimeStampInFile']:
                try:
                    shutil.copyfile(self.sourcedir+"/"+file, self.baddir+file)
                    self.log_writer.log(self.file_object, f"{file} moved to bad data folder due to invalid file name")
                except Exception as e:
                    self.log_writer.log(self.file_object, f"{file} could not be moved to bad data folder with the following error: {e}")
                    raise Exception()
            else:
                try:
                    shutil.copyfile(self.sourcedir+"/"+file, self.gooddir+file)
                    self.log_writer.log(self.file_object, f"{file} moved to good data folder")
                except Exception as e:
                    self.log_writer.log(self.file_object, f"{file} could not be moved to good data folder with the following error: {e}")
                    raise Exception()
        self.log_writer.log(self.file_object, "Finish checking for valid name of files")

    def column_count(self, schema):
        self.log_writer.log(self.file_object, "Start checking for number of columns in file")
        self.schema = schema
        for file in os.listdir(self.gooddir[:-1]):
            try:
                filename = pd.read_csv(os.path.join(self.gooddir,file))
            except Exception as e:
                self.log_writer.log(self.file_object, f"{file} could not be read with the following error: {e}")
                raise Exception()
            if filename.shape[1] != self.schema['NumberofColumns']:
                try:
                    shutil.move(self.gooddir+file, self.baddir+file)
                    self.log_writer.log(self.file_object, f"{file} moved to bad data folder due to mismatch of number of columns")
                except PermissionError:
                    self.log_writer.log(self.file_object, f"{file} is open, please close and try again")
                    raise Exception()
                except OSError:
                    self.log_writer.log(self.file_object, f"{file} file not found")
                    raise Exception()
                except Exception as e:
                    self.log_writer.log(self.file_object, f"{file} file fail to move to bad data folder with the following error: {e}")
                    raise Exception()
        self.log_writer.log(self.file_object, "Finish checking for number of columns in file")
    
    def all_null_column_check(self):
        self.log_writer.log(self.file_object, "Start checking for columns with all missing values in file")
        for file in os.listdir(self.gooddir[:-1]):
            try:
                filename = pd.read_csv(os.path.join(self.gooddir,file))
            except Exception as e:
                self.log_writer.log(self.file_object, f"{file} could not be read with the following error: {e}")
                raise Exception()      
            for column in filename.columns:
                if filename[column].isnull().all():
                    try:
                        shutil.move(self.gooddir+file, self.baddir+file)
                        self.log_writer.log(self.file_object, f"{file} moved to bad data folder due to having columns with all missing values")
                    except PermissionError:
                        self.log_writer.log(self.file_object, f"{file} is open, please close and try again")
                        raise Exception()
                    except OSError:
                        self.log_writer.log(self.file_object, f"{file} file not found")
                        raise Exception()
                    except Exception as e:
                        self.log_writer.log(self.file_object, f"{file} file fail to move to bad data folder with the following error: {e}")
                        raise Exception()
                    break
        self.log_writer.log(self.file_object, "Finish checking for columns with all missing values in file")
    
    def blank_with_null_replacement(self):
        self.log_writer.log(self.file_object, "Start replacing missing values with null keyword")
        for file in os.listdir(self.gooddir[:-1]):
            try:
                filename = pd.read_csv(os.path.join(self.gooddir,file))
                filename.fillna('null', inplace=True)
                filename.to_csv(self.gooddir+file, index=False)
            except Exception as e:
                self.log_writer.log(self.file_object, f"Replacing missing values with null keyword for file {file} fail with the following error: {e}")
                raise Exception()
        self.log_writer.log(self.file_object, "Finish replacing missing values with null keyword")
    
    def remove_temp_good_train_data(self):
        self.log_writer.log(self.file_object, "Start deleting all good_training_data files")
        for file in os.listdir(self.gooddir[:-1]):
            try:
                os.remove(self.gooddir+file)
                self.log_writer.log(self.file_object, f"{file} file deleted from Good_Training_Data folder")
            except PermissionError:
                self.log_writer.log(self.file_object, f"{file} file is open, please close and try again")
                raise Exception()
            except OSError:
                self.log_writer.log(self.file_object, f"{file} file not found")
                raise Exception()
            except Exception as e:
                self.log_writer.log(self.file_object, f"{file} file fail to delete with the following error: {e}")
                raise Exception()
        self.log_writer.log(self.file_object, "Finish deleting all good_training_data files")
    
    def bad_to_archive_data(self, archivedir):
        self.log_writer.log(self.file_object, "Start moving all bad data files into archive folder")
        self.archivedir = archivedir
        for file in os.listdir(self.baddir[:-1]):
            try:
                shutil.move(self.baddir+file, self.archivedir+file)
                self.log_writer.log(self.file_object, f"{file} moved to archive data folder")
            except PermissionError:
                self.log_writer.log(self.file_object, f"{file} is open, please close and try again")
                raise Exception()
            except OSError:
                self.log_writer.log(self.file_object, f"{file} file not found")
                raise Exception()
            except Exception as e:
                self.log_writer.log(self.file_object, f"{file} file fail to move to archive with the following error: {e}")
                raise Exception()
        self.log_writer.log(self.file_object, "Finish moving all bad data files into archive folder")
    
    def initial_data_preparation(self, schemapath, folders, batchfilepath, goodfilepath, archivefilepath, finalfilepath):
        self.log_writer.log(self.file_object, "Start initial data preparation")
        self.schemapath = schemapath
        self.folders = folders
        self.batchfilepath = batchfilepath
        self.goodfilepath = goodfilepath
        self.archivefilepath = archivefilepath
        self.finalfilepath = finalfilepath
        schema = self.load_train_schema(self.schemapath)
        self.file_initialize(self.folders)
        self.file_namecheck(self.batchfilepath,schema)
        self.column_count(schema)
        self.all_null_column_check()
        self.blank_with_null_replacement()
        self.newDB(schema)
        self.data_insert(self.goodfilepath)
        self.compile_data_from_DB(self.finalfilepath)
        self.remove_temp_good_train_data()
        self.bad_to_archive_data(self.archivefilepath)
        self.log_writer.log(self.file_object, "Finish initial data preparation")
