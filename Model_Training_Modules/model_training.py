from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_validate, learning_curve
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.base import clone
import optuna
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.metrics import matthews_corrcoef, recall_score, make_scorer, ConfusionMatrixDisplay, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import numpy as np
from Application_Logger.logger import App_Logger
import pickle as pkl
import matplotlib.pyplot as plt

random_state=42

class model_trainer:
    def __init__(self, file_object):
        self.file_object = file_object
        self.log_writer = App_Logger()
    
    def setting_attributes(trial, cv_results,train_val_mc, test_mc, train_val_recall, test_recall):
        trial.set_user_attr("train_matthews_corrcoef", cv_results['train_matthews_corrcoef'].mean())
        trial.set_user_attr("train_recall_score_macro", cv_results['train_recall_score_macro'].mean())
        trial.set_user_attr("val_matthews_corrcoef", cv_results['test_matthews_corrcoef'].mean())
        trial.set_user_attr("val_recall_score_macro", cv_results['test_recall_score_macro'].mean())
        trial.set_user_attr("train_val_matthews_corrcoef", train_val_mc)
        trial.set_user_attr("test_matthews_corrcoef", test_mc)
        trial.set_user_attr("train_val_recall_score_macro", train_val_recall)
        trial.set_user_attr("test_recall_score_macro", test_recall)
    
    def overfitting_check(cv_results,train_val_recall,test_recall,threshold):
        if np.abs(train_val_recall - test_recall) > threshold:
            raise optuna.TrialPruned()
        matthews_corrcoef = cv_results['test_matthews_corrcoef'].mean()
        recall_score = cv_results['test_recall_score_macro'].mean()
        return matthews_corrcoef, recall_score

    def rf_objective(trial,X_train_data,y_train_data, X_test_data, y_test_data,threshold):
        clf = DecisionTreeClassifier(random_state=random_state)
        path = clf.cost_complexity_pruning_path(X_train_data, y_train_data)
        ccp_alpha = trial.suggest_categorical('ccp_alpha',path.ccp_alphas[:-1])
        n_estimators = trial.suggest_int('n_estimators', 100, 500,1)
        class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample'])
        clf = RandomForestClassifier(random_state=random_state, criterion='gini', n_estimators=n_estimators, 
        class_weight=class_weight, ccp_alpha = ccp_alpha, max_features=None,n_jobs=-1)
        cv_results, train_val_mc, test_mc, train_val_recall, test_recall = model_trainer.classification_metrics(clf,X_train_data,y_train_data,X_test_data, y_test_data)
        matthews_corrcoef, recall_score = model_trainer.overfitting_check(cv_results,train_val_recall,test_recall,threshold)
        model_trainer.setting_attributes(trial,cv_results,train_val_mc, test_mc, train_val_recall, test_recall)
        return matthews_corrcoef,recall_score

    def lr_objective(trial,X_train_data,y_train_data,X_test_data,y_test_data,threshold):
        C = trial.suggest_float('C',0.01,1)
        clf = LogisticRegression(C=C, max_iter=1000000, random_state=random_state, class_weight = 'balanced',n_jobs=-1)
        cv_results, train_val_mc, test_mc, train_val_recall, test_recall = model_trainer.classification_metrics(clf,X_train_data,y_train_data,X_test_data, y_test_data)
        matthews_corrcoef, recall_score = model_trainer.overfitting_check(cv_results,train_val_recall,test_recall,threshold)
        model_trainer.setting_attributes(trial,cv_results,train_val_mc, test_mc, train_val_recall, test_recall)
        return matthews_corrcoef,recall_score

    def dt_objective(trial,X_train_data,y_train_data,X_test_data,y_test_data,threshold):
        clf = DecisionTreeClassifier(random_state=random_state)
        path = clf.cost_complexity_pruning_path(X_train_data, y_train_data)
        ccp_alpha = trial.suggest_categorical('ccp_alpha',path.ccp_alphas[:-1])
        clf = DecisionTreeClassifier(random_state=random_state, criterion='gini', class_weight='balanced', ccp_alpha=ccp_alpha)
        cv_results, train_val_mc, test_mc, train_val_recall, test_recall = model_trainer.classification_metrics(clf,X_train_data,y_train_data,X_test_data, y_test_data)
        matthews_corrcoef, recall_score = model_trainer.overfitting_check(cv_results,train_val_recall,test_recall,threshold)
        model_trainer.setting_attributes(trial,cv_results,train_val_mc, test_mc, train_val_recall, test_recall)
        return matthews_corrcoef,recall_score

    def et_objective(trial,X_train_data,y_train_data,X_test_data,y_test_data,threshold):
        clf = DecisionTreeClassifier(random_state=random_state)
        path = clf.cost_complexity_pruning_path(X_train_data, y_train_data)
        ccp_alpha = trial.suggest_categorical('ccp_alpha',path.ccp_alphas[:-1])
        n_estimators = trial.suggest_int('n_estimators', 100, 500,1)
        class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample'])
        clf = ExtraTreesClassifier(random_state=random_state, n_estimators=n_estimators, criterion='gini', 
        class_weight=class_weight,ccp_alpha=ccp_alpha,max_features=None,n_jobs=-1)
        cv_results, train_val_mc, test_mc, train_val_recall, test_recall = model_trainer.classification_metrics(clf,X_train_data,y_train_data,X_test_data, y_test_data)
        matthews_corrcoef, recall_score = model_trainer.overfitting_check(cv_results,train_val_recall,test_recall,threshold)
        model_trainer.setting_attributes(trial,cv_results,train_val_mc, test_mc, train_val_recall, test_recall)
        return matthews_corrcoef,recall_score

    def svc_objective(trial,X_train_data,y_train_data,X_test_data,y_test_data,threshold):
        C = trial.suggest_float('C',0.01,1)
        clf = SVC(C=C,kernel='linear', probability=True, random_state=random_state, class_weight='balanced')
        cv_results, train_val_mc, test_mc, train_val_recall, test_recall = model_trainer.classification_metrics(clf,X_train_data,y_train_data,X_test_data, y_test_data)
        matthews_corrcoef, recall_score = model_trainer.overfitting_check(cv_results,train_val_recall,test_recall,threshold)
        model_trainer.setting_attributes(trial,cv_results,train_val_mc, test_mc, train_val_recall, test_recall)
        return matthews_corrcoef,recall_score

    def knn_objective(trial,X_train_data,y_train_data,X_test_data,y_test_data,threshold):
        n_neighbors = trial.suggest_categorical('n_neighbors', [3, 5, 7, 9, 11])
        algorithm = trial.suggest_categorical('algorithm', ['ball_tree', 'kd_tree', 'brute'])
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        leaf_size = trial.suggest_int('leaf_size',10,50)
        p = trial.suggest_int('p',1,4)
        clf = KNeighborsClassifier(n_neighbors=n_neighbors,algorithm=algorithm,weights=weights,leaf_size=leaf_size,p=p,n_jobs=-1)
        cv_results, train_val_mc, test_mc, train_val_recall, test_recall = model_trainer.classification_metrics(clf,X_train_data,y_train_data,X_test_data, y_test_data)
        matthews_corrcoef, recall_score = model_trainer.overfitting_check(cv_results,train_val_recall,test_recall,threshold)
        model_trainer.setting_attributes(trial,cv_results,train_val_mc, test_mc, train_val_recall, test_recall)
        return matthews_corrcoef,recall_score

    def gaussiannb_objective(trial,X_train_data,y_train_data,X_test_data,y_test_data,threshold):
        var_smoothing = trial.suggest_float('var_smoothing', 0, 1)
        clf = GaussianNB(var_smoothing=var_smoothing)
        cv_results, train_val_mc, test_mc, train_val_recall, test_recall = model_trainer.classification_metrics(clf,X_train_data,y_train_data,X_test_data, y_test_data)
        matthews_corrcoef, recall_score = model_trainer.overfitting_check(cv_results,train_val_recall,test_recall,threshold)
        model_trainer.setting_attributes(trial,cv_results,train_val_mc, test_mc, train_val_recall, test_recall)
        return matthews_corrcoef,recall_score

    def gradientboost_objective(trial,X_train_data,y_train_data,X_test_data,y_test_data,threshold):
        loss = trial.suggest_categorical('loss',['deviance','exponential'])
        subsample = trial.suggest_float('subsample',0.1,1)
        clf = DecisionTreeClassifier(random_state=random_state)
        path = clf.cost_complexity_pruning_path(X_train_data, y_train_data)
        ccp_alpha = trial.suggest_categorical('ccp_alpha',path.ccp_alphas[:-1])
        learning_rate = trial.suggest_float('learning_rate',0.001,0.01)
        n_estimators = trial.suggest_int('n_estimators', 100, 500,1)
        clf = GradientBoostingClassifier(random_state=random_state, learning_rate=learning_rate,
        n_estimators = n_estimators, loss=loss, ccp_alpha=ccp_alpha,subsample=subsample,max_depth=None)        
        cv_results, train_val_mc, test_mc, train_val_recall, test_recall = model_trainer.classification_metrics(clf,X_train_data,y_train_data,X_test_data, y_test_data)
        matthews_corrcoef, recall_score = model_trainer.overfitting_check(cv_results,train_val_recall,test_recall,threshold)
        model_trainer.setting_attributes(trial,cv_results,train_val_mc, test_mc, train_val_recall, test_recall)
        return matthews_corrcoef,recall_score

    def adaboost_objective(trial,X_train_data,y_train_data,X_test_data,y_test_data,threshold):
        learning_rate = trial.suggest_float('learning_rate',0.001,0.01)
        n_estimators = trial.suggest_int('n_estimators', 100, 500,1)
        clf = AdaBoostClassifier(learning_rate=learning_rate, n_estimators=n_estimators, random_state=random_state)
        cv_results, train_val_mc, test_mc, train_val_recall, test_recall = model_trainer.classification_metrics(clf,X_train_data,y_train_data,X_test_data, y_test_data)
        matthews_corrcoef, recall_score = model_trainer.overfitting_check(cv_results,train_val_recall,test_recall,threshold)
        model_trainer.setting_attributes(trial,cv_results,train_val_mc, test_mc, train_val_recall, test_recall)
        return matthews_corrcoef,recall_score

    def xgboost_objective(trial,X_train_data,y_train_data,X_test_data,y_test_data,threshold):
        booster = trial.suggest_categorical('booster',['gbtree','gblinear'])
        eta = trial.suggest_float('eta',0,0.01)
        gamma = trial.suggest_float('gamma',1,50)
        min_child_weight = trial.suggest_float('min_child_weight',1,50)
        max_delta_step = trial.suggest_int('max_delta_step',1,10)
        colsample_bytree = trial.suggest_float('colsample_bytree',0,1)
        colsample_bylevel = trial.suggest_float('colsample_bylevel',0,1)
        colsample_bynode = trial.suggest_float('colsample_bynode',0,1)
        lambdas = trial.suggest_float('lambda',1,2)
        alpha = trial.suggest_float('alpha',0,1)
        sample_type = trial.suggest_categorical('sample_type',['uniform','weighted'])
        normalize_type = trial.suggest_categorical('normalize_type',['tree','forest'])
        n_estimators = trial.suggest_int('n_estimators', 100, 500,1)
        clf = XGBClassifier(objective='binary:logistic', eval_metric='aucpr', verbosity=0, booster=booster,
        eta=eta, gamma=gamma, min_child_weight=min_child_weight, max_delta_step=max_delta_step, subsample=0.5, 
        colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel, colsample_bynode=colsample_bynode, 
        lambdas=lambdas, alpha=alpha, sample_type=sample_type, normalize_type=normalize_type, random_state=random_state, 
        n_estimators=n_estimators,max_depth=None)
        cv_results, train_val_mc, test_mc, train_val_recall, test_recall = model_trainer.classification_metrics(clf,X_train_data,y_train_data,X_test_data, y_test_data)
        matthews_corrcoef, recall_score = model_trainer.overfitting_check(cv_results,train_val_recall,test_recall,threshold)
        model_trainer.setting_attributes(trial,cv_results,train_val_mc, test_mc, train_val_recall, test_recall)
        return matthews_corrcoef,recall_score
        
    def classification_metrics(clf,X_train_data,y_train_data, X_test_data, y_test_data):
        cv_results = cross_validate(clf, X_train_data, y_train_data, cv=5, return_train_score=True,
        scoring={"matthews_corrcoef": make_scorer(matthews_corrcoef), "recall_score_macro": make_scorer(recall_score, average='macro')})
        clf.fit(X_train_data,y_train_data)
        train_val_mc = matthews_corrcoef(y_train_data,clf.predict(X_train_data))
        test_mc = matthews_corrcoef(y_test_data,clf.predict(X_test_data))
        train_val_recall = recall_score(y_train_data,clf.predict(X_train_data),average='macro')
        test_recall = recall_score(y_test_data,clf.predict(X_test_data),average='macro')
        return cv_results, train_val_mc, test_mc, train_val_recall, test_recall

    def initialize_model_training(self,folderpath,filepath):
        self.log_writer.log(self.file_object, 'Start initializing objectives required for model training')
        self.filepath = filepath
        objectives = [model_trainer.rf_objective, model_trainer.lr_objective, model_trainer.dt_objective, 
        model_trainer.et_objective, model_trainer.svc_objective, model_trainer.knn_objective, model_trainer.gaussiannb_objective,
        model_trainer.adaboost_objective, model_trainer.gradientboost_objective, model_trainer.xgboost_objective]
        selectors = [RandomForestClassifier(random_state=random_state,max_features=None,criterion='gini',n_jobs=-1), 
        LogisticRegression(max_iter=1000000,random_state=random_state,class_weight = 'balanced',n_jobs=-1), 
        DecisionTreeClassifier(random_state=random_state,criterion='gini',class_weight = 'balanced'), 
        ExtraTreesClassifier(random_state=random_state,max_features=None,criterion='gini',n_jobs=-1), 
        SVC(probability=True, random_state=random_state,kernel='linear', class_weight='balanced'), 
        KNeighborsClassifier(n_jobs=-1), GaussianNB(), AdaBoostClassifier(random_state=random_state), 
        GradientBoostingClassifier(random_state=random_state,max_depth=None), 
        XGBClassifier(objective='binary:logistic', subsample=0.5, eval_metric='aucpr', verbosity=0, random_state=random_state,max_depth=None)]
        try:
            results = pd.concat([pd.Series(name='column_list'), pd.Series(name='num_features'), pd.Series(name='model_name'), 
            pd.Series(name='best_params'), pd.Series(name='clustering_indicator'),pd.Series(name='train_matthews_corrcoef'), 
            pd.Series(name='val_matthews_corrcoef'), pd.Series(name='train_recall_score'),pd.Series(name='val_recall_score'), 
            pd.Series(name='train_val_matthews_corrcoef'), pd.Series(name='test_matthews_corrcoef'), 
            pd.Series(name='train_val_recall_score'),pd.Series(name='test_recall_score')], axis=1)
            results.to_csv(folderpath+filepath, mode='w',index=False)
        except Exception as e:
            self.log_writer.log(self.file_object, f'Fail to create initial csv file of results from model training with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, 'Finish initializing objectives required for model training')
        return objectives, selectors

    def fit_scaled_data(self, data, scaler_type):
        self.log_writer.log(self.file_object, f'Start fitting scaler on dataset')
        try:
            scaler_type = scaler_type.fit(data)
        except Exception as e:
            self.log_writer.log(self.file_object, f'Fitting scaler on dataset failed with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish fitting scaler on dataset')
        return scaler_type

    def transform_scaled_data(self, data, scaler_type):
        self.log_writer.log(self.file_object, f'Start transforming scaler on dataset')
        try:
            data_scaled = pd.DataFrame(scaler_type.transform(data), columns=data.columns)
        except Exception as e:
            self.log_writer.log(self.file_object, f'Transforming scaler on dataset failed with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish transforming scaler on dataset')
        return data_scaled

    def scale_vs_non_scale_data(self,clf,X_train_scaled,X_train,y_train):
        if type(clf).__name__ in ['LogisticRegression','SVC','KNeighborsClassifier']:
            X_train_data = X_train_scaled
        else:
            X_train_data = X_train
        y_train_data = y_train
        X_train_data = X_train_data.reset_index(drop=True)
        y_train_data = y_train_data.reset_index(drop=True)
        return X_train_data, y_train_data

    def optuna_optimizer(self, obj, n_trials):
        self.log_writer.log(self.file_object, f'Start performing optuna hyperparameter tuning for {obj.__name__} model')
        try:
            study = optuna.create_study(directions=['maximize','maximize'])
            study.optimize(obj, n_trials=n_trials,n_jobs=-1)
            trials = study.best_trials
        except Exception as e:
            self.log_writer.log(self.file_object, f'Performing optuna hyperparameter tuning for {obj.__name__} model failed with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish performing optuna hyperparameter tuning for {obj.__name__} model')
        return trials

    def train_per_model(self,obj, clf, trial_size,col_list, n_features,X_train,y_train,X_test,y_test,num_features, col_selected, model_name, best_params, clustering_yes_no, train_matthews_corrcoef, val_matthews_corrcoef, train_recall_score, val_recall_score, train_val_matthews_corrcoef_score, test_matthews_corrcoef_score, train_val_recall_score, test_recall_score, clustering,threshold):
        self.log_writer.log(self.file_object, f'Start model training on {type(clf).__name__} for {n_features} features with {clustering} clustering')
        try:
            func = lambda trial: obj(trial, X_train, y_train,X_test,y_test,threshold)
            func.__name__ = type(clf).__name__
            model = clone(clf)
            trials = self.optuna_optimizer(func,trial_size)
            for trial in trials:
                if trial.user_attrs['train_val_matthews_corrcoef'] <= 0 or trial.user_attrs['test_matthews_corrcoef'] <= 0:
                    self.log_writer.log(self.file_object, f'Model training on {type(clf).__name__} for {n_features} features with {clustering} clustering for trial {trial.number} is not considered, since its overall performance is worse than a random prediction model')
                    continue
                num_features.append(n_features)
                col_selected.append(col_list)
                model_name.append(type(model).__name__)
                best_params.append(model.set_params(**trial.params).get_params())
                clustering_yes_no.append(clustering)
                train_matthews_corrcoef.append(trial.user_attrs['train_matthews_corrcoef'])
                val_matthews_corrcoef.append(trial.user_attrs['val_matthews_corrcoef'])
                train_recall_score.append(trial.user_attrs['train_recall_score_macro'])
                val_recall_score.append(trial.user_attrs['val_recall_score_macro'])
                train_val_matthews_corrcoef_score.append(trial.user_attrs['train_val_matthews_corrcoef'])
                train_val_recall_score.append(trial.user_attrs['train_val_recall_score_macro'])                
                test_matthews_corrcoef_score.append(trial.user_attrs['test_matthews_corrcoef'])
                test_recall_score.append(trial.user_attrs['test_recall_score_macro']) 
                self.log_writer.log(self.file_object, f"Results for {type(model).__name__} with {n_features} features and {clustering} clustering saved for trial {trial.number}")
        except Exception as e:
            self.log_writer.log(self.file_object, f'Model training on {type(clf).__name__} for {n_features} features with {clustering} clustering failed with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish model training on {type(clf).__name__} for {n_features} features with {clustering} clustering')

    def store_tuning_results(self,col_selected,num_features,model_name,best_params,clustering_yes_no, train_matthews_corrcoef,val_matthews_corrcoef,train_recall_score,val_recall_score,train_val_matthews_corrcoef_score,test_matthews_corrcoef_score,train_val_recall_score,test_recall_score,folderpath,filepath,n_features):
        self.log_writer.log(self.file_object, f'Start appending results from model training for {n_features} features')
        try:
            results = pd.concat([pd.Series(col_selected, name='column_list'), pd.Series(num_features, name='num_features'), 
                                pd.Series(model_name, name='model_name'), pd.Series(best_params, name='best_params'),
                                pd.Series(clustering_yes_no, name='clustering_indicator'), 
                                pd.Series(train_matthews_corrcoef, name='train_matthews_corrcoef'), 
                                pd.Series(val_matthews_corrcoef, name='val_matthews_corrcoef'), 
                                pd.Series(train_recall_score, name='train_recall_score'), pd.Series(val_recall_score, name='val_recall_score'),
                                pd.Series(train_val_matthews_corrcoef_score, name='train_val_matthews_corrcoef'),
                                pd.Series(test_matthews_corrcoef_score, name='test_matthews_corrcoef'),
                                pd.Series(train_val_recall_score, name='train_val_recall_score'),
                                pd.Series(test_recall_score, name='test_recall_score'),], axis=1)
            results.to_csv(folderpath+filepath, mode='a',header=False, index=False)
        except Exception as e:
            self.log_writer.log(self.file_object, f'Appending results from model training for {n_features} features failed with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish appending results from model training for {n_features} features')

    def best_model(self, folderpath, filepath, bestresultpath):
        self.log_writer.log(self.file_object, f'Start determining best configuration to use for saving models')
        try:
            results = pd.read_csv(folderpath+filepath).sort_values(by='test_matthews_corrcoef',ascending=False)
            final_models = results[(results['test_matthews_corrcoef'] == results['test_matthews_corrcoef'].max()) & (results['test_recall_score'] == results['test_recall_score'].max())].sort_values(by=['num_features','clustering_indicator'])
            # If no model performs best for both metrics, then pick the model with the highest recall score on the test set
            if len(final_models) == 0:
                final_models = results[(results['test_recall_score'] == results['test_recall_score'].max())].sort_values(by=['num_features','clustering_indicator'])
            pd.DataFrame(final_models, columns = final_models.columns).to_csv(folderpath+bestresultpath,index=False)
        except Exception as e:
            self.log_writer.log(self.file_object, f'Fail to determine best configuration to use for saving models with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish determining best configuration to use for saving models')
        return final_models

    def k_means_clustering(self, data, start_cluster, end_cluster):
        self.log_writer.log(self.file_object, f'Start deriving best number of clusters from k-means clustering')
        wcss=[]
        try:
            for i in range (start_cluster,end_cluster+1):
                kmeans=KMeans(n_clusters=i,init='k-means++',random_state=random_state)
                kmeans.fit(data)
                wcss.append(kmeans.inertia_)
            kneeloc = KneeLocator(range(start_cluster,end_cluster+1), wcss, curve='convex', direction='decreasing')
            kmeans=KMeans(n_clusters=kneeloc.knee,init='k-means++',random_state=random_state)
        except Exception as e:
            self.log_writer.log(self.file_object, f'Deriving best number of clusters from k-means clustering failed with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish deriving best number of clusters from k-means clustering')
        return kneeloc, kmeans

    def add_cluster_number_to_data(self,train_scaled_data, train_data, test_scaled_data, test_data, final=False):
        self.log_writer.log(self.file_object, f'Start performing data clustering')
        try:
            kneeloc, kmeans = self.k_means_clustering(train_scaled_data, 1, 10)
            train_scaled_data['cluster'] = kmeans.fit_predict(train_scaled_data)
            train_data['cluster'] = train_scaled_data['cluster']
            test_scaled_data['cluster'] = kmeans.predict(test_scaled_data)
            test_data['cluster'] = test_scaled_data['cluster']
            train_scaled_data.reset_index(drop=True, inplace=True)
            train_data.reset_index(drop=True, inplace=True)
            test_scaled_data.reset_index(drop=True, inplace=True)
            test_data.reset_index(drop=True, inplace=True)
            if final == True:
                pkl.dump(kmeans, open('Saved_Models/kmeans_model.pkl', 'wb'))
        except Exception as e:
            self.log_writer.log(self.file_object, f'Performing data clustering failed with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish performing data clustering that contains {kneeloc.knee} clusters')
        return train_data, test_data, train_scaled_data, test_scaled_data

    def data_scaling_train_test(self, folderpath, train_data, gaussian_variables, test_data, non_gaussian_variables):
        self.log_writer.log(self.file_object, f'Start performing scaling data on train and test set')
        try:
            non_gaussian_absolute = []
            non_gaussian = []
            for variable in non_gaussian_variables:
                if (train_data[variable]<0).sum() == 0:
                    non_gaussian_absolute.append(variable)
                else:
                    non_gaussian.append(variable)
            scaler = self.fit_scaled_data(train_data[gaussian_variables], StandardScaler())
            gaussian_train_data_scaled = self.transform_scaled_data(train_data[gaussian_variables], scaler)
            gaussian_test_data_scaled = self.transform_scaled_data(test_data[gaussian_variables], scaler)
            pkl.dump(scaler, open(folderpath+'StandardScaler.pkl', 'wb'))
            scaler = self.fit_scaled_data(train_data[non_gaussian], MinMaxScaler())
            non_gaussian_train_data_scaled = self.transform_scaled_data(train_data[non_gaussian], scaler)
            non_gaussian_test_data_scaled = self.transform_scaled_data(test_data[non_gaussian], scaler)
            pkl.dump(scaler, open(folderpath+'MinMaxScaler.pkl', 'wb'))
            scaler = self.fit_scaled_data(train_data[non_gaussian_absolute], MaxAbsScaler())
            non_gaussian_absolute_train_data_scaled = self.transform_scaled_data(train_data[non_gaussian_absolute], scaler)
            non_gaussian_absolute_test_data_scaled = self.transform_scaled_data(test_data[non_gaussian_absolute], scaler)
            pkl.dump(scaler, open(folderpath+'MaxAbsScaler.pkl', 'wb'))
            X_train_scaled = pd.concat([gaussian_train_data_scaled, non_gaussian_train_data_scaled,non_gaussian_absolute_train_data_scaled], axis=1)
            X_test_scaled = pd.concat([gaussian_test_data_scaled, non_gaussian_test_data_scaled,non_gaussian_absolute_test_data_scaled], axis=1)
        except Exception as e:
            self.log_writer.log(self.file_object, f'Data scaling on training and test set failed with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish performing scaling data on train and test set')
        return X_train_scaled, X_test_scaled

    def learning_curve_plot(self,folderpath, train_size, train_score_m, test_score_m):
        fig1, ax1 = plt.subplots()
        ax1.plot(train_size, train_score_m, 'o-', color="b")
        ax1.plot(train_size, test_score_m, 'o-', color="r")
        ax1.legend(('Training score', 'Test score'), loc='best')
        ax1.set_xlabel("Training Samples")
        ax1.set_ylabel("recall score")
        ax1.set_title("Learning Curve Analysis (CV=5)")
        ax1.grid()
        ax1.annotate(np.round(train_score_m[-1],4),(train_size[-1]-20,train_score_m[-1]+0.015))
        ax1.annotate(np.round(test_score_m[-1],4),(train_size[-1]-20,test_score_m[-1]-0.015))
        plt.savefig(folderpath+'Learning_Curve_Analysis.png')

    def train_overall_model(self, X_train_sub, X_test_sub, y_train_data, y_test_data, model, final_result, name_model, folderpath):
        self.log_writer.log(self.file_object, f'Start training and saving the {name_model} model')
        try:
            X_sub = pd.concat([X_train_sub, X_test_sub]).reset_index(drop=True)
            y_sub = pd.concat([y_train_data, y_test_data]).reset_index(drop=True)
            overall_model = clone(model)
            overall_model.set_params(**eval(final_result['best_params'].values[0].replace("\'missing\': nan,","").replace("'", "\"")))
            overall_model.fit(X_sub,y_sub)
            ConfusionMatrixDisplay.from_predictions(y_sub, overall_model.predict(X_sub)).plot()
            plt.title('Overall confusion matrix')
            plt.savefig(folderpath+'Overall_Confusion_Matrix.png')
            report = classification_report(y_sub, overall_model.predict(X_sub), output_dict=True)
            pd.DataFrame(report).transpose().to_csv(folderpath + name_model + '_Classification_Report.csv')
            train_size, train_score, test_score = learning_curve(estimator=overall_model, X=X_sub, y=y_sub, cv=5, scoring=make_scorer(recall_score, average='macro'))
            train_score_m = np.mean(np.abs(train_score), axis=1)
            test_score_m = np.mean(np.abs(test_score), axis=1)
            self.learning_curve_plot(folderpath, train_size, train_score_m, test_score_m)
        except Exception as e:
            self.log_writer.log(self.file_object, f'Training and saving the {name_model} model failed with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish training and saving the {name_model} model')
        return overall_model

    def train_model_and_hyperparameter_tuning(self, train_data, test_data, train_output, test_output, folderpath, filepath, bestresultpath, threshold):
        self.log_writer.log(self.file_object, 'Start model training and hyperparameter tuning')
        self.train_data = train_data
        self.test_data = test_data
        self.train_output = train_output
        self.test_output = test_output
        self.folderpath = folderpath
        self.filepath = filepath
        self.bestresultpath = bestresultpath
        self.threshold = threshold
        optuna.logging.set_verbosity(optuna.logging.DEBUG)
        objectives, selectors = self.initialize_model_training(self.folderpath,self.filepath)
        gaussian_variables = list(pd.read_csv(self.folderpath+'Gaussian_columns.csv')['Variable'])
        non_gaussian_variables = list(pd.read_csv(self.folderpath+'Non_gaussian_columns.csv')['Variable'])
        X_train_scaled, X_test_scaled = self.data_scaling_train_test(self.folderpath, self.train_data, gaussian_variables, self.test_data, non_gaussian_variables)
        X_train_scaled = X_train_scaled[self.train_data.columns]
        X_test_scaled = X_test_scaled[self.test_data.columns]

        for n_features in range(1,min(21,len(X_train_scaled.columns)+1)):
            num_features, col_selected, model_name, best_params, clustering_yes_no  = [], [], [], [], []
            train_matthews_corrcoef, val_matthews_corrcoef, train_recall_score, val_recall_score = [], [], [], []
            train_val_matthews_corrcoef_score, test_matthews_corrcoef_score, train_val_recall_score, test_recall_score = [], [], [], []
            for obj, clf in zip(objectives, selectors):
                X_train_data, y_train_data = self.scale_vs_non_scale_data(clf,X_train_scaled,self.train_data,self.train_output)
                X_test_data, y_test_data = self.scale_vs_non_scale_data(clf,X_test_scaled,self.test_data,self.test_output)                       
                transformer = SelectKBest(f_classif, k=n_features)
                X_train_sub = pd.DataFrame(transformer.fit_transform(X_train_data, y_train_data), columns=transformer.get_feature_names_out())
                X_test_sub= pd.DataFrame(transformer.transform(X_test_data), columns=transformer.get_feature_names_out())
                col_list = list(transformer.get_feature_names_out())
                for clustering in ['yes', 'no']:
                    col_list = list(transformer.get_feature_names_out())
                    if clustering == 'yes':
                        X_train_sub_temp = X_train_sub.copy()
                        X_test_sub_temp = X_test_sub.copy()
                        X_train_scaled_sub = pd.DataFrame(transformer.transform(X_train_scaled), columns = col_list)
                        X_test_scaled_sub = pd.DataFrame(transformer.transform(X_test_scaled), columns = col_list)
                        X_train_cluster_sub, X_test_cluster_sub, X_train_scaled_cluster_sub, X_test_scaled_cluster_sub = self.add_cluster_number_to_data(X_train_scaled_sub, X_train_sub_temp, X_test_scaled_sub, X_test_sub_temp)
                        X_train_cluster_data, y_train_data = self.scale_vs_non_scale_data(clf,X_train_scaled_cluster_sub,X_train_cluster_sub,y_train_data)
                        X_test_cluster_data, y_test_data = self.scale_vs_non_scale_data(clf,X_test_scaled_cluster_sub,X_test_cluster_sub,y_test_data) 
                        col_list.extend(pd.get_dummies(X_train_cluster_data['cluster'], drop_first=True).columns.tolist())
                        X_train_cluster_data = pd.concat([X_train_cluster_data.iloc[:,:-1], pd.get_dummies(X_train_cluster_data['cluster'], drop_first=True)],axis=1)
                        X_test_cluster_data = pd.concat([X_test_cluster_data.iloc[:,:-1], pd.get_dummies(X_test_cluster_data['cluster'], drop_first=True)],axis=1)
                        self.train_per_model(obj, clf, 50, col_list, n_features,X_train_cluster_data,y_train_data,X_test_cluster_data,
                        y_test_data,num_features, col_selected, model_name, best_params, clustering_yes_no,
                        train_matthews_corrcoef, val_matthews_corrcoef, train_recall_score, val_recall_score, train_val_matthews_corrcoef_score, 
                        test_matthews_corrcoef_score, train_val_recall_score, test_recall_score, clustering,self.threshold)
                    else:
                        self.train_per_model(obj, clf, 50, col_list, n_features,X_train_sub,y_train_data,X_test_sub,
                        y_test_data,num_features, col_selected, model_name, best_params, clustering_yes_no,
                        train_matthews_corrcoef, val_matthews_corrcoef, train_recall_score, val_recall_score, train_val_matthews_corrcoef_score, 
                        test_matthews_corrcoef_score, train_val_recall_score, test_recall_score, clustering,self.threshold)                            
            self.store_tuning_results(col_selected,num_features,model_name,best_params, clustering_yes_no, 
            train_matthews_corrcoef,val_matthews_corrcoef,train_recall_score, val_recall_score, train_val_matthews_corrcoef_score,test_matthews_corrcoef_score,
            train_val_recall_score, test_recall_score, self.folderpath, self.filepath,n_features)
        
        final_result = self.best_model(self.folderpath, self.filepath, self.bestresultpath)
        name_model = final_result['model_name'].values[0]
        num_features = final_result['num_features'].values[0]
        clustering = final_result['clustering_indicator'].values[0]
        model_dict = {'RandomForestClassifier':RandomForestClassifier(),'LogisticRegression':LogisticRegression(),
        'DecisionTreeClassifier':DecisionTreeClassifier(),'SVC':SVC(),'KNeighborsClassifier':KNeighborsClassifier(),
        'GaussianNB':GaussianNB(),'XGBClassifier':XGBClassifier(),'ExtraTreesClassifier':ExtraTreesClassifier(),
        'AdaBoostClassifier':AdaBoostClassifier(),'GradientBoostingClassifier':GradientBoostingClassifier()}
        model = model_dict[name_model]
        X_train_data, y_train_data = self.scale_vs_non_scale_data(model,X_train_scaled,self.train_data,self.train_output)
        X_test_data, y_test_data = self.scale_vs_non_scale_data(model,X_test_scaled,self.test_data,self.test_output)
        columns = final_result['column_list'].values[0].replace("'","").strip("][").split(', ')[:num_features]
        X_train_sub = X_train_data[columns]
        X_test_sub = X_test_data[columns]

        if clustering == 'yes':
            X_train_scaled_sub = X_train_scaled[columns]
            X_test_scaled_sub = X_test_scaled[columns]
            X_train_cluster_sub, X_test_cluster_sub, X_train_scaled_cluster_sub, X_test_scaled_cluster_sub = self.add_cluster_number_to_data(X_train_scaled_sub, X_train_sub, X_test_scaled_sub, X_test_sub, final=True)
            X_train_cluster_data, y_train_data = self.scale_vs_non_scale_data(model,X_train_scaled_cluster_sub,X_train_cluster_sub,y_train_data)
            X_test_cluster_data, y_test_data = self.scale_vs_non_scale_data(model,X_test_scaled_cluster_sub,X_test_cluster_sub,y_test_data) 
            X_train_sub = pd.concat([X_train_cluster_data.iloc[:,:-1], pd.get_dummies(X_train_cluster_data['cluster'], drop_first=True)],axis=1)
            X_test_sub = pd.concat([X_test_cluster_data.iloc[:,:-1], pd.get_dummies(X_test_cluster_data['cluster'], drop_first=True)],axis=1)
        trained_model = self.train_overall_model(X_train_sub, X_test_sub, y_train_data, y_test_data, model, final_result, name_model, self.folderpath)
        pkl.dump(trained_model,open('Saved_Models/'+name_model+'_'+clustering+'_clustering.pkl','wb'))
        self.log_writer.log(self.file_object, 'Finish model training and hyperparameter tuning')