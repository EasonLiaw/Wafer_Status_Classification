'''
Author: Liaw Yi Xian
Last Modified: 20th June 2022
'''
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from Application_Logger.logger import App_Logger
import pickle as pkl

class pred_Preprocessor:
    def __init__(self, file_object):
        '''
            Method Name: __init__
            Description: This method initializes instance of Preprocessor class
            Output: None
        '''
        self.file_object = file_object
        self.log_writer = App_Logger()

    def extract_compiled_data(self, path):
        '''
            Method Name: extract_compiled_data
            Description: This method extracts data from a csv file and converts it into a pandas dataframe.
            Output: A pandas dataframe
            On Failure: Raise Exception
        '''
        self.log_writer.log(self.file_object, "Start reading compiled data from database")
        self.path = path
        try:
            data = pd.read_csv(path)
        except Exception as e:
            self.log_writer.log(self.file_object, f"Fail to read compiled data from database with the following error: {e}")
            raise Exception(f"Fail to read compiled data from database with the following error: {e}")
        self.log_writer.log(self.file_object, "Finish reading compiled data from database")
        return data

    def remove_duplicated_rows(self, data):
        '''
            Method Name: remove_duplicated_rows
            Description: This method removes duplicated rows from a pandas dataframe.
            Output: A pandas DataFrame after removing duplicated rows. In addition, duplicated records that are removed will 
            be stored in a separate csv file labeled "Duplicated_Records_Removed.csv"
            On Failure: Raise Exception
        '''
        self.log_writer.log(self.file_object, "Start handling duplicated rows in the dataset")
        self.data = data
        if len(self.data[self.data.duplicated()]) == 0:
            self.log_writer.log(self.file_object, "No duplicated rows found in the dataset")
        else:
            try:
                self.data[self.data.duplicated()].to_csv('Intermediate_Pred_Results/Duplicated_Records_Removed.csv', index=False)
                self.data = self.data.drop_duplicates(ignore_index=True)
            except Exception as e:
                self.log_writer.log(self.file_object, f"Fail to remove duplicated rows with the following error: {e}")
                raise Exception(f"Fail to remove duplicated rows with the following error: {e}")
        self.log_writer.log(self.file_object, "Finish handling duplicated rows in the dataset")
        return self.data

    def impute_missing_values(self, X, column_list, method):
        '''
            Method Name: impute_missing_values
            Description: This method imputes missing values based on classified method from classify_impute_method function.
            Output: A pandas dataframe after imputing missing values.
            On Failure: Raise Exception
        '''
        self.log_writer.log(self.file_object, "Start imputing missing values in the dataset")
        try:
            if len(column_list)>0:
                if method == 'mean':
                    imputer = pkl.load(open('Intermediate_Train_Results/SimpleMeanImputer.pkl', 'rb'))
                    X[column_list] = imputer.transform(X[column_list])
                    self.log_writer.log(self.file_object, f"The following columns have been imputed using simple mean strategy: {column_list}")
                elif method == 'median':
                    imputer = pkl.load(open('Intermediate_Train_Results/SimpleMedianImputer.pkl', 'rb'))
                    X[column_list] = imputer.transform(X[column_list])
                    self.log_writer.log(self.file_object, f"The following columns have been imputed using simple median strategy: {column_list}")
                elif method == 'iterative':
                    imputer = pkl.load(open('Intermediate_Train_Results/IterativeImputer.pkl', 'rb'))
                    X = pd.DataFrame(imputer.transform(X), columns=X.columns)
                    self.log_writer.log(self.file_object, f"The following columns have been imputed using iterative strategy: {column_list}")
        except Exception as e:
            self.log_writer.log(self.file_object, f"Fail to impute missing values with the following error: {e}")
            raise Exception(f"Fail to impute missing values with the following error: {e}")
        return X

    def outlier_capping(self, method, X):
        '''
            Method Name: outlier_capping
            Description: This method caps outliers identified at lower bound and upper bound values obtained from
            iqr_lower_upper_bound or gaussian_lower_upper_bound function for non-gaussian variables and gaussian variables
            respectively.
            Output: A pandas dataframe, where outlier values are capped at lower bound/upper bound.
            On Failure: Raise Exception
        '''
        self.log_writer.log(self.file_object, f"Start capping outliers using {method} method")
        try:
            if method == 'iqr':
                winsorizer = pkl.load(open('Intermediate_Train_Results/iqr_outliercapping.pkl', 'rb'))
            elif method == 'gaussian':
                winsorizer = pkl.load(open('Intermediate_Train_Results/gaussian_outliercapping.pkl', 'rb'))
            X = winsorizer.transform(X)
        except Exception as e:
            self.log_writer.log(self.file_object, f"Fail to cap outliers using {method} method with the following error: {e}")
            raise Exception(f"Fail to cap outliers using {method} method with the following error: {e}")
        self.log_writer.log(self.file_object, f"Finish capping outliers using {method} method") 
        return X

    def drop_constant_variance(self, X):
        '''
            Method Name: drop_constant_variance
            Description: This method removes variables that have constant variance from the dataset.
            Output: A pandas dataframe, where variables with constant variance are removed.  In addition, variables
            that were removed due to constant variance are stored in a csv file named as "Columns_Removed.csv"
            (One csv file for gaussian transformed data and another csv file for non gaussian transformed data)
            On Failure: Raise Exception
        '''
        self.log_writer.log(self.file_object, "Start removing features with constant variance")
        try:
            selector = pkl.load(open('Intermediate_Train_Results/dropconstantvariance.pkl', 'rb'))
            X = selector.transform(X)
            self.log_writer.log(self.file_object, f"Following set of features were removed due to having constant variance: {selector.features_to_drop_}")
        except Exception as e:
            self.log_writer.log(self.file_object, f"Fail to remove features with constant variance with the following error: {e}")
            raise Exception(f"Fail to remove features with constant variance with the following error: {e}")
        self.log_writer.log(self.file_object, "Finish removing features with constant variance")
        return X

    def gaussian_transform(self, X, best_results):
        '''
            Method Name: gaussian_transform
            Description: This method transforms individual variables that are identified to be significant for gaussian 
            transformation using identified methods from gaussian_transform_test function.
            Output: A pandas dataframe, where relevant non-gaussian variables are transformed into gaussian variables
            On Failure: Raise Exception
        '''
        self.log_writer.log(self.file_object, f"Start transforming non gaussian columns to gaussian columns")
        try:
            transformer_types = best_results['Transformation_Type'].unique()
            for type in transformer_types:
                variable_list = list(best_results[best_results['Transformation_Type'] == type]['Variable'])
                if type == 'logarithmic':
                    transformer = pkl.load(open('Intermediate_Train_Results/logarithmic_transformation.pkl', 'rb'))
                elif type == 'reciprocal':
                    transformer = pkl.load(open('Intermediate_Train_Results/reciprocal_transformation.pkl', 'rb'))
                elif type == 'square-root':
                    transformer = pkl.load(open('Intermediate_Train_Results/square-root_transformation.pkl', 'rb'))
                elif type == 'yeo-johnson':
                    transformer = pkl.load(open('Intermediate_Train_Results/yeo-johnson_transformation.pkl', 'rb'))
                elif type == 'square':
                    transformer = pkl.load(open('Intermediate_Train_Results/square_transformation.pkl', 'rb'))
                X = transformer.transform(X)
                self.log_writer.log(self.file_object, f'{type} transformation is applied to {variable_list} columns')
        except Exception as e:
            self.log_writer.log(self.file_object, f"Fail to transform non-gaussian variables into gaussian variables with the following error: {e}")
            raise Exception(f"Fail to transform non-gaussian variables into gaussian variables with the following error: {e}")
        self.log_writer.log(self.file_object, f"Finish transforming non gaussian columns to gaussian columns")
        return X
    
    def data_preprocessing(self, start_path, best_result):
        '''
            Method Name: data_preprocessing
            Description: This method performs all the data preprocessing tasks for the data.
            Output: A pandas dataframe, where all the data preprocessing tasks are performed.
            On Failure: Raise Exception
        '''
        self.log_writer.log(self.file_object, 'Start of data preprocessing')
        self.start_path = start_path
        self.best_result = best_result
        best_result_df = pd.read_csv(self.best_result,index_col=False)
        num_features = best_result_df['num_features'].values[0]
        column_list = best_result_df['column_list'].values[0].replace("'","").strip("][").split(', ')[:num_features]
        data = self.extract_compiled_data(self.start_path)
        index = data['Wafer']
        data = self.remove_duplicated_rows(data)
        initial_columns_to_remove = pd.read_csv('Intermediate_Train_Results/Columns_Drop_from_Original.csv',index_col=False)
        initial_columns_to_remove = initial_columns_to_remove[(initial_columns_to_remove['Reason'] == 'Irrelevant column') | (initial_columns_to_remove['Reason'] == 'More than 80% missing values')]['Columns_Removed'].tolist()
        data = data.drop(initial_columns_to_remove,axis=1)
        imputation_df = pd.read_csv('Intermediate_Train_Results/Imputation_Methods.csv',index_col=False)
        mean_imp_col = imputation_df[imputation_df['Impute_Strategy'] == 'mean']['Column_Name'].tolist()
        median_imp_col = imputation_df[imputation_df['Impute_Strategy'] == 'median']['Column_Name'].tolist()
        iterative_imp_col = imputation_df[imputation_df['Impute_Strategy'] == 'iterative']['Column_Name'].tolist()
        data = self.impute_missing_values(data, mean_imp_col, 'mean')
        data = self.impute_missing_values(data, median_imp_col, 'median')
        data = self.impute_missing_values(data, iterative_imp_col, 'iterative')
        data = self.outlier_capping('iqr', data)
        data = self.outlier_capping('gaussian', data)
        data = self.drop_constant_variance(data)
        data = self.gaussian_transform(data, pd.read_csv('Intermediate_Train_Results/Best_Transformation_Non_Gaussian.csv',index_col=False))
        data = pd.concat([index,data],axis=1)
        self.log_writer.log(self.file_object, 'End of data preprocessing')
        return data, column_list

