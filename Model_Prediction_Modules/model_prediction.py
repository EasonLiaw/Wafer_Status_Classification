import pandas as pd
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from Application_Logger.logger import App_Logger

class model_predictor:
    def __init__(self, file_object):
        self.file_object = file_object
        self.log_writer = App_Logger()
    
    def clustering(self, clustermodelpath, data, index):
        self.log_writer.log(self.file_object, 'Start using best clustering model based on best configurations identified for prediction')
        try:
            indexer = data[index]
            features = data.iloc[:,1:]
            scaler = StandardScaler()
            data_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
            kmeans = pkl.load(open(clustermodelpath,'rb'))
            features['cluster'] = kmeans.predict(data_scaled)
            features = pd.concat([features.iloc[:,:-1], pd.get_dummies(features['cluster'], drop_first=True)],axis=1)
            data = pd.concat([indexer, features],axis=1)
        except Exception as e:
            self.log_writer.log(self.file_object, 'Fail to make predictions from best clustering model based on best configurations identified with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, 'Finish making predictions using best clustering model based on best configurations identified')
        return data
    
    def model_prediction(self, bestresultpath, modelobject, clustermodelobject, data, index):
        self.log_writer.log(self.file_object, 'Start storing best models for model deployment')
        self.bestresultpath = bestresultpath
        self.modelobject = modelobject
        self.clustermodelobject = clustermodelobject
        self.data = data
        self.index = index
        try:
            best_result = pd.read_csv(self.bestresultpath)
            clustering = best_result['clustering_indicator'].values[0]
            if clustering == 'yes':
                combined_data = self.clustering(self.clustermodelobject, self.data, self.index)
            else:
                combined_data = self.data
            indexer = combined_data[self.index]
            X = combined_data.drop([self.index], axis=1)
            name_model = best_result['model_name'].values[0]
            if name_model in ['LogisticRegression','SVC','KNeighborsClassifier']:
                scaler = StandardScaler()
                X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            with open(self.modelobject,'rb') as model:
                model_for_pred = pkl.load(model)
                X['Pred_Output'] = model_for_pred.predict(X)
            X = pd.concat([indexer, X],axis=1)
            X.to_csv('Intermediate_Pred_Results/Predictions.csv',index=False)
            self.log_writer.log(self.file_object, f'{name_model} model used to make predictions successfully')
        except Exception as e:
            self.log_writer.log(self.file_object, f'Fail to make prediction using best models for model deployment with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, 'Finish making predictions using best models for model deployment')