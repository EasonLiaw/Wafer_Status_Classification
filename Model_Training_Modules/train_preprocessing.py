import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from Application_Logger.logger import App_Logger
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import scipy.stats as st
from sklearn.model_selection import train_test_split
import pickle as pkl
import feature_engine.imputation as fei
import feature_engine.selection as fes
import feature_engine.outliers as feo
import feature_engine.transformation as fet

class train_Preprocessor:
    def __init__(self, file_object, result_dir):
        '''
            Method Name: __init__
            Description: This method initializes instance of Preprocessor class
            Output: None
            On Failure: Raise Exception

            Written By: Yi Xian Liaw
            Version: 1.0
            Revisions: None
        '''
        self.file_object = file_object
        self.result_dir = result_dir
        self.log_writer = App_Logger()

    def extract_compiled_data(self, path):
        '''
            Method Name: extract_compiled_data
            Description: This method extracts data from a csv file and converts it into a pandas dataframe.
            Output: A pandas dataframe
            On Failure: Raise Exception

            Written By: Yi Xian Liaw
            Version: 1.0
            Revisions: None
        '''
        self.log_writer.log(self.file_object, "Start reading compiled data from database")
        self.path = path
        try:
            data = pd.read_csv(path)
        except Exception as e:
            self.log_writer.log(self.file_object, f"Fail to read compiled data from database with the following error: {e}")
            raise Exception()
        self.log_writer.log(self.file_object, "Finish reading compiled data from database")
        return data

    def remove_irrelevant_columns(self, data, column):
        '''
            Method Name: remove_irrelevant_columns
            Description: This method removes columns from a pandas dataframe, which are not relevant for analysis.
            Output: A pandas DataFrame after removing the specified columns. In addition, columns that are removed will be 
            stored in a separate csv file labeled "Columns_Removed.csv" (One csv file for gaussian transformed data and 
            another csv file for non gaussian transformed data)
            On Failure: Raise Exception

            Written By: Yi Xian Liaw
            Version: 1.0
            Revisions: None
        '''
        self.log_writer.log(self.file_object, "Start removing irrelevant columns from the dataset")
        try:
            data = data.drop(column, axis=1)
            result = pd.concat([pd.Series(column, name='Columns_Removed'), pd.Series(["Irrelevant column"]*len([column]), name='Reason')], axis=1)
            result.to_csv(self.result_dir+'Columns_Drop_from_Original.csv', index=False)
        except Exception as e:
            self.log_writer.log(self.file_object, f"Irrelevant columns could not be removed from the dataset with the following error: {e}")
            raise Exception()
        self.log_writer.log(self.file_object, "Finish removing irrelevant columns from the dataset")
        return data

    def remove_duplicated_rows(self, data):
        '''
            Method Name: remove_duplicated_rows
            Description: This method removes duplicated rows from a pandas dataframe.
            Output: A pandas DataFrame after removing duplicated rows. In addition, duplicated records that are removed will 
            be stored in a separate csv file labeled "Duplicated_Records_Removed.csv"
            On Failure: Raise Exception

            Written By: Yi Xian Liaw
            Version: 1.0
            Revisions: None
        '''
        self.log_writer.log(self.file_object, "Start handling duplicated rows in the dataset")
        if len(data[data.duplicated()]) == 0:
            self.log_writer.log(self.file_object, "No duplicated rows found in the dataset")
        else:
            try:
                data[data.duplicated()].to_csv(self.result_dir+'Duplicated_Records_Removed.csv', index=False)
                data = data.drop_duplicates(ignore_index=True)
            except Exception as e:
                self.log_writer.log(self.file_object, f"Fail to remove duplicated rows with the following error: {e}")
                raise Exception()
        self.log_writer.log(self.file_object, "Finish handling duplicated rows in the dataset")
        return data
    
    def features_and_labels(self,data,column):
        '''
            Method Name: features_and_labels
            Description: This method splits a pandas dataframe into two pandas objects, consist of features and target labels.
            Output: Two pandas/series objects consist of features and labels separately.
            On Failure: Raise Exception

            Written By: Yi Xian Liaw
            Version: 1.0
            Revisions: None
        '''
        self.log_writer.log(self.file_object, "Start separating the data into features and labels")
        try:
            X = data.drop(column, axis=1)
            y = data[column]
        except Exception as e:
            self.log_writer.log(self.file_object, f"Fail to separate features and labels with the following error: {e}")
            raise Exception()
        self.log_writer.log(self.file_object, "Finish separating the data into features and labels")
        return X, y

    def classify_impute_method(self, X_train, X_test):
        '''
            Method Name: classify_impute_method
            Description: This method classifies imputation method used to handle missing values for different columns.
            Output: Four list of columns for every imputation method (Mean, Median, Iterative Mean, Iterative Median)
            On Failure: Raise Exception

            Written By: Yi Xian Liaw
            Version: 1.0
            Revisions: None
        '''
        self.log_writer.log(self.file_object, "Start identifying methods for handling missing values in the dataset")
        mean_imputed_column, median_imputed_column, iterative_imputed_column = [], [], []
        try:
            result = pd.concat([pd.Series(X_train.isna().sum(), name='Number_Missing_Values'),pd.Series(X_train.isna().sum()/len(X_train), name='Proportion_Missing_Values')], axis=1)
            result.to_csv(self.result_dir+'Missing_Values_Info.csv')
            result = X_train[X_train.isnull().any(axis=1)]
            result.to_csv(self.result_dir+'Missing_Values_Records.csv', index=False)
        except Exception as e:
            self.log_writer.log(self.file_object, f"Fail to store information related to missing values with the following error: {e}")
            raise Exception()
        try:
            for column in X_train.columns:
                if X_train[column].isna().sum()/len(X_train)>0.8:
                    X_train = X_train.drop(column, axis=1)
                    X_test = X_test.drop(column, axis=1)
                    result = pd.concat([pd.Series(column, name='Columns_Removed'), pd.Series("More than 80% missing values", name='Reason')], axis=1)
                    result.to_csv(self.result_dir+'Columns_Drop_from_Original.csv', index=False, header=False, mode='a+')
                elif X_train[column].isna().sum()/len(X_train)>0:
                    # For data that is missing completely at random (very weak to weak correlation of missingness with other features)
                    if (np.abs(X_train.isnull().corr()[column]).dropna()>0.6).sum()-1 == 0:
                        if (X_train[column].skew()>-0.5) & (X_train[column].skew()<0.5):
                            mean_imputed_column.append(column)
                            self.log_writer.log(self.file_object, f"Missing values in {column} column will be handled using simple mean imputation")
                        else:
                            median_imputed_column.append(column)
                            self.log_writer.log(self.file_object, f"Missing values in {column} column will be handled using simple median imputation")
                    # For data that is missing at random (strong to very strong correlation of missingness with other features)
                    else:
                        iterative_imputed_column.append(column)
                        self.log_writer.log(self.file_object, f"Missing values in {column} column will be handled using iterative imputation")
                        
            pd.concat([pd.DataFrame({'Column_Name': mean_imputed_column,'Impute_Strategy':['mean']*len(mean_imputed_column)}),
            pd.DataFrame({'Column_Name': median_imputed_column,'Impute_Strategy':['median']*len(median_imputed_column)}),
            pd.DataFrame({'Column_Name': iterative_imputed_column,'Impute_Strategy':['iterative']*len(iterative_imputed_column)})]).to_csv(self.result_dir+'Imputation_Methods.csv', index=False)
        except Exception as e:
            self.log_writer.log(self.file_object, f"Fail to identify imputation methods with the following error: {e}")
            raise Exception()
        self.log_writer.log(self.file_object, "Finish identifying methods for handling missing values in the dataset")
        return mean_imputed_column, median_imputed_column, iterative_imputed_column, X_train, X_test

    def impute_missing_values(self, X_train, X_test, column_list, method):
        '''
            Method Name: impute_missing_values
            Description: This method imputes missing values based on classified method from classify_impute_method function.
            Output: A pandas dataframe after imputing missing values.
            On Failure: Raise Exception

            Written By: Yi Xian Liaw
            Version: 1.0
            Revisions: None
        '''
        self.log_writer.log(self.file_object, "Start imputing missing values in the dataset")
        try:
            if len(column_list)>0:
                if method == 'mean':
                    imputer = fei.MeanMedianImputer('mean')
                    X_train[column_list] = imputer.fit_transform(X_train[column_list])
                    X_test[column_list] = imputer.transform(X_test[column_list])
                    pkl.dump(imputer, open(self.result_dir+'SimpleMeanImputer.pkl', 'wb'))
                    self.log_writer.log(self.file_object, f"The following columns have been imputed using simple mean strategy: {column_list}")
                elif method == 'median':
                    imputer = fei.MeanMedianImputer('median')
                    X_train[column_list] = imputer.fit_transform(X_train[column_list])
                    X_test[column_list] = imputer.transform(X_test[column_list])
                    pkl.dump(imputer, open(self.result_dir+'SimpleMedianImputer.pkl','wb'))
                    self.log_writer.log(self.file_object, f"The following columns have been imputed using simple median strategy: {column_list}")
                elif method == 'iterative':
                    imputer = IterativeImputer(max_iter = 10, verbose=1, initial_strategy='mean', random_state=42, 
                    n_nearest_features=min(20, len(X_train.columns)))
                    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
                    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
                    pkl.dump(imputer, open(self.result_dir+'IterativeImputer.pkl','wb'))
                    self.log_writer.log(self.file_object, f"The following columns have been imputed using iterative strategy: {column_list}")
        except Exception as e:
            self.log_writer.log(self.file_object, f"Fail to impute missing values with the following error: {e}")
            raise Exception()
        return X_train, X_test
    
    def check_gaussian(self, X_train):
        '''
            Method Name: check_gaussian
            Description: This method classifies columns into gaussian and non-gaussian columns based on anderson test.
            Output: Two list of columns for gaussian and non-gaussian variables.
            On Failure: Raise Exception

            Written By: Yi Xian Liaw
            Version: 1.0
            Revisions: None
        '''
        self.log_writer.log(self.file_object, "Start categorizing columns into gaussian vs non-gaussian distribution")
        gaussian_columns = []
        non_gaussian_columns = []
        try:
            for column in X_train.columns:
                result = st.anderson(X_train[column])
                if result[0] > result[1][2]:
                    non_gaussian_columns.append(column)
                    self.log_writer.log(self.file_object, f"{column} column is identified as non-gaussian")
                else:
                    gaussian_columns.append(column)
                    self.log_writer.log(self.file_object, f"{column} column is identified as gaussian")
        except Exception as e:
            self.log_writer.log(self.file_object, f"Fail to categorize columns into gaussian vs non-gaussian distribution with the following error: {e}")
            raise Exception()
        self.log_writer.log(self.file_object, "Finish categorizing columns into gaussian vs non-gaussian distribution")
        return gaussian_columns, non_gaussian_columns

    def iqr_lower_upper_bound(self, X_train, column):
        '''
            Method Name: iqr_lower_upper_bound
            Description: This method computes lower bound and upper bound of outliers based on interquartile range (IQR) method
            Output: Two floating values that consist of lower bound and upper bound of outlier points for non-gaussian variables
            On Failure: Raise Exception

            Written By: Yi Xian Liaw
            Version: 1.0
            Revisions: None
        '''
        self.log_writer.log(self.file_object, f"Start computing lower and upper bound of outliers for {column} column")
        try:
            Q1 = X_train[column].quantile(0.25)
            Q3 = X_train[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        except Exception as e:
            self.log_writer.log(self.file_object, f"Fail to compute lower and upper bound of outliers for {column} column with the following error: {e}")
            raise Exception()
        self.log_writer.log(self.file_object, f"Finish computing lower and upper bound of outliers for {column} column")
        return lower_bound, upper_bound

    def gaussian_lower_upper_bound(self, X_train, column):
        '''
            Method Name: gaussian_lower_upper_bound
            Description: This method computes lower bound and upper bound of outliers based on gaussian method
            Output: Two floating values that consist of lower bound and upper bound of outlier points for gaussian variables
            On Failure: Raise Exception

            Written By: Yi Xian Liaw
            Version: 1.0
            Revisions: None
        '''
        self.log_writer.log(self.file_object, f"Start computing lower and upper bound of outliers for {column} column")
        try:
            lower_bound = np.mean(X_train[column]) - 3 * np.std(X_train[column])
            upper_bound = np.mean(X_train[column]) + 3 * np.std(X_train[column])
        except Exception as e:
            self.log_writer.log(self.file_object, f"Fail to compute lower and upper bound of outliers for {column} column with the following error: {e}")
            raise Exception()
        self.log_writer.log(self.file_object, f"Finish computing lower and upper bound of outliers for {column} column")
        return lower_bound, upper_bound

    def check_outliers(self, X_train, column_list, type):
        '''
            Method Name: check_outliers
            Description: This method computes number and proportion of outliers for every variable based on type of variable
            (gaussian vs non-gaussian) that is categorized from check_gaussian function.
            Output: No output returned. Instead, the results that contains number and proportion of outliers for every variable
            are stored in a csv file named as "Outliers_Info.csv" 
            (One csv file for gaussian variables and another csv file for non gaussian variables)
            On Failure: Raise Exception

            Written By: Yi Xian Liaw
            Version: 1.0
            Revisions: None
        '''
        self.log_writer.log(self.file_object, f"Start checking outliers for the following columns: {column_list}") 
        outlier_num = []
        outlier_prop = []
        try:
            for column in column_list:
                if type == 'non-gaussian':
                    lower_bound, upper_bound = self.iqr_lower_upper_bound(X_train, column)
                elif type == 'gaussian':
                    lower_bound, upper_bound = self.gaussian_lower_upper_bound(X_train, column)
                outlier_num.append(len(X_train[(X_train[column] < lower_bound) | (X_train[column] > upper_bound)]))
                outlier_prop.append(np.round(len(X_train[(X_train[column] < lower_bound) | (X_train[column] > upper_bound)])/len(X_train),4))
            results = pd.concat([pd.Series(X_train[column_list].columns, name='Variable'),pd.Series(outlier_num,name='Number_Outliers'),pd.Series(outlier_prop, name='Prop_Outliers')],axis=1)
            if type == 'non-gaussian':
                results.to_csv(self.result_dir + 'Outliers_Info_Non_Gaussian.csv', index=False)
            elif type == 'gaussian':
                results.to_csv(self.result_dir + 'Outliers_Info_Gaussian.csv', index=False)
        except Exception as e:
            self.log_writer.log(self.file_object, f"Fail to check outliers for the following columns: {column_list} with the following error: {e}")
            raise Exception()
        self.log_writer.log(self.file_object, f"Finish checking outliers for the following columns: {column_list}") 

    def outlier_capping(self, method, fold, X_train, X_test, column_list):
        '''
            Method Name: outlier_capping
            Description: This method caps outliers identified at lower bound and upper bound values obtained from
            iqr_lower_upper_bound or gaussian_lower_upper_bound function for non-gaussian variables and gaussian variables
            respectively.
            Output: A pandas dataframe, where outlier values are capped at lower bound/upper bound.
            On Failure: Raise Exception

            Written By: Yi Xian Liaw
            Version: 1.0
            Revisions: None
        '''
        self.log_writer.log(self.file_object, f"Start capping outliers for the following columns: {column_list}")
        try:
            winsorizer = feo.Winsorizer(capping_method=method, tail='both', fold=fold, add_indicators=False,variables=column_list)
            X_train = winsorizer.fit_transform(X_train)
            X_test = winsorizer.transform(X_test)
            pkl.dump(winsorizer, open(self.result_dir + f'{method}_outliercapping.pkl','wb'))
        except Exception as e:
            self.log_writer.log(self.file_object, f"Fail to cap outliers for the following columns: {column_list} with the following error: {e}")
            raise Exception()
        self.log_writer.log(self.file_object, f"Finish capping outliers for the following columns: {column_list}") 
        return X_train, X_test
    
    def drop_constant_variance(self, X_train, X_test, filename):
        '''
            Method Name: drop_constant_variance
            Description: This method removes variables that have constant variance from the dataset.
            Output: A pandas dataframe, where variables with constant variance are removed.  In addition, variables
            that were removed due to constant variance are stored in a csv file named as "Columns_Removed.csv"
            (One csv file for gaussian transformed data and another csv file for non gaussian transformed data)
            On Failure: Raise Exception

            Written By: Yi Xian Liaw
            Version: 1.0
            Revisions: None
        '''
        self.log_writer.log(self.file_object, "Start removing features with constant variance")
        try:
            selector = fes.DropConstantFeatures()
            X_train = selector.fit_transform(X_train)
            X_test = selector.transform(X_test)
            pkl.dump(selector, open(self.result_dir+'dropconstantvariance.pkl', 'wb'))
            result = pd.concat([pd.Series(selector.features_to_drop_, name='Columns_Removed'), pd.Series(["Constant variance"]*len(selector.features_to_drop_), name='Reason')], axis=1)
            result.to_csv(self.result_dir + filename, index=False, mode='a+', header=False)
            self.log_writer.log(self.file_object, f"Following set of features were removed due to having constant variance: {selector.features_to_drop_}")
        except Exception as e:
            self.log_writer.log(self.file_object, f"Fail to remove features with constant variance with the following error: {e}")
            raise Exception()
        self.log_writer.log(self.file_object, "Finish removing features with constant variance")
        return X_train, X_test

    def gaussian_transform(self, X_train, X_test, best_results):
        '''
            Method Name: gaussian_transform
            Description: This method transforms individual variables that are identified to be significant for gaussian 
            transformation using identified methods from gaussian_transform_test function.
            Output: A pandas dataframe, where relevant non-gaussian variables are transformed into gaussian variables
            On Failure: Raise Exception

            Written By: Yi Xian Liaw
            Version: 1.0
            Revisions: None
        '''
        X_train_transformed = X_train.copy()
        X_test_transformed = X_test.copy()
        try:
            transformer_types = best_results['Transformation_Type'].unique()
            for type in transformer_types:
                variable_list = list(best_results[best_results['Transformation_Type'] == type].index)
                if type == 'logarithmic':
                    transformer = fet.LogTransformer(variables=variable_list)
                elif type == 'reciprocal':
                    transformer = fet.ReciprocalTransformer(variables=variable_list)
                elif type == 'square-root':
                    transformer = fet.PowerTransformer(exp=0.5, variables=variable_list)
                elif type == 'yeo-johnson':
                    transformer = fet.YeoJohnsonTransformer(variables=variable_list)
                elif type == 'square':
                    transformer = fet.PowerTransformer(exp=2, variables=variable_list)
                X_train_transformed = transformer.fit_transform(X_train_transformed)
                X_test_transformed = transformer.transform(X_test_transformed)
                pkl.dump(transformer, open(self.result_dir+f'{type}_transformation.pkl','wb'))
                self.log_writer.log(self.file_object, f'{type} transformation is applied to {variable_list} columns')
        except Exception as e:
            self.log_writer.log(self.file_object, f"Fail to transform non-gaussian variables into gaussian variables with the following error: {e}")
            raise Exception()
        return X_train_transformed, X_test_transformed

    def gaussian_transform_test(self, X_train, X_test, column_list):
        '''
            Method Name: gaussian_transform_test
            Description: This method performs the following gaussian transformation methods (Logarithmic, Exponential, Square, 
            Square-root, Yeo-Johnson) for different non-gaussian variables and its significance for transforming non-gaussian to 
            gaussian variable is identified using anderson test. After obtaining best transformation techniques for different variables,
            gaussian transformation is performed on the dataset using gaussian_transform function.
            Output: A pandas dataframe, where variables with constant variance are removed. In addition, best results identified 
            that transforms non-gaussian to gaussian variables are stored in a csv file named as 
            "Best_Transformation_Non_Gaussian.csv"
            On Failure: Raise Exception

            Written By: Yi Xian Liaw
            Version: 1.0
            Revisions: None
        '''
        self.log_writer.log(self.file_object, 'Start testing for gaussian transformation on non-gaussian columns')
        try:
            transformer_list = [fet.LogTransformer(), fet.ReciprocalTransformer(), fet.PowerTransformer(exp=0.5),fet.YeoJohnsonTransformer(), fet.PowerTransformer(exp=2)]
            transformer_names = ['logarithmic','reciprocal','square-root','yeo-johnson','square']
            result_names, result_test_stats, result_skewness, result_kurtosis, result_columns, result_critical_value=[], [], [], [], [], []
            for transformer, name in zip(transformer_list, transformer_names):
                for column in column_list:
                    try:
                        X_transformed = transformer.fit_transform(X_train[[column]])
                        X_test_transformed = transformer.transform(X_test[[column]])
                        result_columns.append(column)
                        result_names.append(name)
                        result_test_stats.append(st.anderson(X_transformed[column])[0])
                        result_critical_value.append(st.anderson(X_transformed[column])[1][2])
                        result_skewness.append(np.round(X_transformed[column].skew(),6))
                        result_kurtosis.append(np.round(X_transformed[column].kurt(),6))
                    except Exception as e:
                        self.log_writer.log(self.file_object, f'{transformer} on {column} column has the following error: {e}')
                        continue
            results = pd.DataFrame([pd.Series(result_columns, name='Variable'), pd.Series(result_names,name='Transformation_Type'),
            pd.Series(result_test_stats, name='Test-stats'), pd.Series(result_critical_value, name='Critical value'),
            pd.Series(result_skewness, name='Skewness'),pd.Series(result_kurtosis,name='Kurtosis')]).T
            best_results = results[results['Test-stats']<results['Critical value']].groupby(by='Variable')[['Transformation_Type','Test-stats']].min()
            best_results.to_csv(self.result_dir + 'Best_Transformation_Non_Gaussian.csv')
            X_train_transformed, X_test_transformed = self.gaussian_transform(X_train, X_test, best_results)
        except Exception as e:
            self.log_writer.log(self.file_object, f"Fail to perform gaussian transformation on non-gaussian columns with the following error: {e}")
            raise Exception()
        self.log_writer.log(self.file_object, 'Finish testing for gaussian transformation on non-gaussian columns')
        return X_train_transformed, X_test_transformed

    def remove_strong_correlated_features(self, X_train, X_test, filename):
        '''
            Method Name: remove_strong_correlated_features
            Description: This method removes features that are very strongly correlated (more than 0.8) with each other.
            Output: A pandas dataframe, where variables that are very strongly correlated with each other are removed.
            In addition, variables that were removed due to very strong correlation with other features are stored in a csv file 
            named as "Columns_Removed.csv" (One csv file for gaussian transformed data and another csv file for 
            non gaussian transformed data) 
            On Failure: Raise Exception

            Written By: Yi Xian Liaw
            Version: 1.0
            Revisions: None
        '''
        self.log_writer.log(self.file_object, "Start removing highly correlated features with one another")
        try:
            cor_remover = fes.DropCorrelatedFeatures(threshold=0.8)
            X_train = cor_remover.fit_transform(X_train)
            X_test = cor_remover.transform(X_test)
            result = pd.concat([pd.Series(list(cor_remover.features_to_drop_), name='Columns_Removed'), pd.Series(["Highly correlated with other features"]*len(list(cor_remover.features_to_drop_)), name='Reason')], axis=1)
            result.to_csv(self.result_dir + filename, index=False, mode='a+', header=False)
            self.log_writer.log(self.file_object, f"Following set of features were removed due to having very high correlation with other features: {cor_remover.features_to_drop_}")
        except Exception as e:
            self.log_writer.log(self.file_object, f"Fail to remove features that are highly correlated with other features with the following error: {e}")
            raise Exception()
        self.log_writer.log(self.file_object, "Finish removing highly correlated features with one another")
        return X_train, X_test
    
    def data_preprocessing(self, start_path, end_path_for_transform, col_remove, target_col):
        '''
            Method Name: data_preprocessing
            Description: This method performs all the data preprocessing tasks for the data.
            Output: A pandas dataframe, where all the data preprocessing tasks are performed.
            On Failure: Raise Exception

            Written By: Raghav Pal
            Version: 1.0
            Revisions: None
        '''
        self.log_writer.log(self.file_object, 'Start of data preprocessing')
        self.start_path = start_path
        self.end_path_for_transform = end_path_for_transform
        self.col_remove = col_remove
        self.target_col = target_col
        data = self.extract_compiled_data(self.start_path)
        data = self.remove_irrelevant_columns(data, self.col_remove)
        data = self.remove_duplicated_rows(data)
        X, y = self.features_and_labels(data, self.target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        mean_imp_col, median_imp_col, iterative_imp_col, X_train, X_test = self.classify_impute_method(X_train, X_test)
        X_train, X_test = self.impute_missing_values(X_train, X_test, mean_imp_col, 'mean')
        X_train, X_test = self.impute_missing_values(X_train, X_test, median_imp_col, 'median')
        X_train, X_test = self.impute_missing_values(X_train, X_test, iterative_imp_col, 'iterative')
        gaussian_columns, non_gaussian_columns = self.check_gaussian(X_train)
        self.check_outliers(X_train, non_gaussian_columns, 'non-gaussian')
        self.check_outliers(X_train, gaussian_columns, 'gaussian')
        X_train, X_test = self.outlier_capping('iqr', 1.5, X_train, X_test, non_gaussian_columns)
        X_train, X_test = self.outlier_capping('gaussian', 3, X_train, X_test, gaussian_columns)
        X_train, X_test = self.drop_constant_variance(X_train, X_test, self.end_path_for_transform)
        gaussian_columns, non_gaussian_columns = self.check_gaussian(X_train)
        X_train_transformed, X_test_transformed = self.gaussian_transform_test(X_train, X_test, non_gaussian_columns)
        X_train_transformed, X_test_transformed = self.remove_strong_correlated_features(X_train_transformed, X_test_transformed, self.end_path_for_transform)
        gaussian_columns, non_gaussian_columns = self.check_gaussian(X_train_transformed)
        pd.Series(gaussian_columns, name='Variable').to_csv(self.result_dir+'Gaussian_columns.csv',index=False)
        pd.Series(non_gaussian_columns, name='Variable').to_csv(self.result_dir+'Non_gaussian_columns.csv',index=False)
        X_train_transformed.to_csv(self.result_dir+'X_train.csv',index=False)
        X_test_transformed.to_csv(self.result_dir+'X_test.csv',index=False)
        y_train.to_csv(self.result_dir+'y_train.csv',index=False)
        y_test.to_csv(self.result_dir+'y_test.csv',index=False)
        self.log_writer.log(self.file_object, 'End of data preprocessing')
        return X_train_transformed, X_test_transformed, y_train, y_test

