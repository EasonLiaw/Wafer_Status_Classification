import pandas as pd
import pickle as pkl
from Application_Logger.logger import App_Logger

class model_predictor:
    def __init__(self, file_object):
        self.file_object = file_object
        self.log_writer = App_Logger()
    
    def clustering(self, clustermodelpath, data, index, features):
        self.log_writer.log(self.file_object, 'Start using best clustering model based on best configurations identified for prediction')
        try:
            indexer = data[index]
            features_df = data[features]
            data_scaled = self.data_scaling_train_test(features_df)
            data_scaled = data_scaled[features]
            kmeans = pkl.load(open(clustermodelpath,'rb'))
            features_df['cluster'] = kmeans.predict(data_scaled)
            features_df = pd.concat([features_df.iloc[:,:-1], pd.get_dummies(features_df['cluster'], drop_first=True)],axis=1)
            clustered_data = pd.concat([indexer, features_df],axis=1)
        except Exception as e:
            self.log_writer.log(self.file_object, f'Fail to make predictions from best clustering model based on best configurations identified with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, 'Finish making predictions using best clustering model based on best configurations identified')
        return clustered_data
    
    def data_scaling_train_test(self,data):
        gaussian_variables = list(pd.read_csv('Intermediate_Train_Results/Gaussian_columns.csv')['Variable'])
        non_gaussian_variables = list(pd.read_csv('Intermediate_Train_Results/Non_gaussian_columns.csv')['Variable'])
        non_gaussian_absolute = []
        non_gaussian = []
        for variable in non_gaussian_variables:
            if (data[variable]<0).sum() == 0:
                non_gaussian_absolute.append(variable)
            else:
                non_gaussian.append(variable)
        scaler = pkl.load(open('Intermediate_Train_Results/StandardScaler.pkl','rb'))
        gaussian_data_scaled = pd.DataFrame(scaler.transform(data[gaussian_variables]), columns=gaussian_variables)
        scaler = pkl.load(open('Intermediate_Train_Results/MinMaxScaler.pkl','rb'))
        non_gaussian_data_scaled = pd.DataFrame(scaler.transform(data[non_gaussian]), columns=non_gaussian)
        scaler = pkl.load(open('Intermediate_Train_Results/MaxAbsScaler.pkl','rb'))
        non_gaussian_absolute_data_scaled = pd.DataFrame(scaler.transform(data[non_gaussian_absolute]), columns=non_gaussian_absolute)
        scaled_data = pd.concat([gaussian_data_scaled, non_gaussian_data_scaled,non_gaussian_absolute_data_scaled], axis=1)
        return scaled_data
    
    def model_prediction(self, bestresultpath, modelobject, clustermodelobject, data, index, column_list):
        self.log_writer.log(self.file_object, 'Start storing best models for model deployment')
        self.bestresultpath = bestresultpath
        self.modelobject = modelobject
        self.clustermodelobject = clustermodelobject
        self.data = data
        self.index = index
        self.column_list = column_list
        try:
            best_result = pd.read_csv(self.bestresultpath)
            clustering = best_result['clustering_indicator'].values[0]
            if clustering == 'yes':
                combined_data = self.clustering(self.clustermodelobject, self.data, self.index, self.column_list)
            else:
                combined_data = self.data
            indexer = combined_data[self.index]
            X = combined_data.drop([self.index], axis=1)
            name_model = best_result['model_name'].values[0]
            if name_model in ['LogisticRegression','SVC','KNeighborsClassifier']:
                X = self.data_scaling_train_test(X)
            X_sub = X[self.column_list]
            with open(self.modelobject,'rb') as model:
                model_for_pred = pkl.load(model)
                X_sub['Pred_Output'] = model_for_pred.predict(X_sub)
            X_sub = pd.concat([indexer, X_sub],axis=1)
            X_sub.to_csv('Intermediate_Pred_Results/Predictions.csv',index=False)
            self.log_writer.log(self.file_object, f'{name_model} model used to make predictions successfully')
        except Exception as e:
            self.log_writer.log(self.file_object, f'Fail to make prediction using best models for model deployment with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, 'Finish making predictions using best models for model deployment')