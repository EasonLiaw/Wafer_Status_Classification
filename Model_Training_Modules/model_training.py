from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, cross_validate, learning_curve
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.base import clone
import optuna
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import matthews_corrcoef, fbeta_score, make_scorer, ConfusionMatrixDisplay, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.combine import SMOTETomek
import pandas as pd
import numpy as np
from Application_Logger.logger import App_Logger
import pickle as pkl
import matplotlib.pyplot as plt

class model_trainer:
    def __init__(self, file_object):
        self.file_object = file_object
        self.log_writer = App_Logger()
    
    def rf_objective(trial,X_train_data,y_train_data):
        ccp_alpha = trial.suggest_float('ccp_alpha', 0,1)
        criterion = trial.suggest_categorical('criterion',['gini','entropy'])
        n_estimators = trial.suggest_int('n_estimators', 50, 200,1)
        class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
        clf = RandomForestClassifier(random_state=42,ccp_alpha = ccp_alpha, criterion=criterion, n_estimators=n_estimators, class_weight=class_weight)
        matthews_corrcoef,f2_score = model_trainer.classification_metrics(clf,X_train_data,y_train_data)
        return matthews_corrcoef,f2_score

    def lr_objective(trial,X_train_data,y_train_data):
        C = trial.suggest_float('C',0,2)
        class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
        clf = LogisticRegression(C=C, max_iter=1000000, random_state=42, class_weight = class_weight)
        matthews_corrcoef,f2_score = model_trainer.classification_metrics(clf,X_train_data,y_train_data)
        return matthews_corrcoef,f2_score

    def dt_objective(trial,X_train_data,y_train_data):
        clf = DecisionTreeClassifier(random_state=42)
        path = clf.cost_complexity_pruning_path(X_train_data,y_train_data)
        ccp_alphas = path.ccp_alphas
        ccp_alpha = trial.suggest_categorical('ccp_alpha', ccp_alphas)
        criterion = trial.suggest_categorical('criterion',['gini','entropy'])
        class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
        clf = DecisionTreeClassifier(random_state=42, ccp_alpha = ccp_alpha, criterion=criterion, class_weight=class_weight)
        matthews_corrcoef,f2_score = model_trainer.classification_metrics(clf,X_train_data,y_train_data)
        return matthews_corrcoef,f2_score

    def et_objective(trial,X_train_data,y_train_data):
        ccp_alpha = trial.suggest_float('ccp_alpha', 0, 1)
        criterion = trial.suggest_categorical('criterion',['gini','entropy'])
        n_estimators = trial.suggest_int('n_estimators', 50, 500,1)
        class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
        clf = ExtraTreesClassifier(random_state=42, ccp_alpha = ccp_alpha, n_estimators=n_estimators, criterion=criterion, class_weight=class_weight)
        matthews_corrcoef,f2_score = model_trainer.classification_metrics(clf,X_train_data,y_train_data)
        return matthews_corrcoef,f2_score

    def svc_objective(trial,X_train_data,y_train_data):
        C = trial.suggest_float('C',0.1,2)
        class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
        clf = SVC(C=C,kernel='linear',probability=True, random_state=42, class_weight=class_weight)
        matthews_corrcoef,f2_score = model_trainer.classification_metrics(clf,X_train_data,y_train_data)
        return matthews_corrcoef,f2_score

    def knn_objective(trial,X_train_data,y_train_data):
        n_neighbors = trial.suggest_categorical('n_neighbors', [3, 5, 7, 9, 11])
        algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        leaf_size = trial.suggest_int('leaf_size',10,100)
        p = trial.suggest_int('p',1,5)
        clf = KNeighborsClassifier(n_neighbors=n_neighbors,algorithm=algorithm,weights=weights,leaf_size=leaf_size,p=p)
        matthews_corrcoef,f2_score = model_trainer.classification_metrics(clf,X_train_data,y_train_data)
        return matthews_corrcoef,f2_score

    def gaussiannb_objective(trial,X_train_data,y_train_data):
        var_smoothing = trial.suggest_float('var_smoothing', 0, 1)
        clf = GaussianNB(var_smoothing=var_smoothing)
        matthews_corrcoef,f2_score = model_trainer.classification_metrics(clf,X_train_data,y_train_data)
        return matthews_corrcoef,f2_score

    def gradientboost_objective(trial,X_train_data,y_train_data):
        loss = trial.suggest_categorical('loss',['deviance','exponential'])
        ccp_alpha = trial.suggest_float('ccp_alpha', 0, 1)
        learning_rate = trial.suggest_float('learning_rate',0.0001,0.2)
        n_estimators = trial.suggest_int('n_estimators', 50, 500,1)
        clf = GradientBoostingClassifier(random_state=42, ccp_alpha = ccp_alpha, learning_rate=learning_rate, 
        n_estimators = n_estimators, max_depth=None, loss=loss)        
        matthews_corrcoef,f2_score = model_trainer.classification_metrics(clf,X_train_data,y_train_data)
        return matthews_corrcoef,f2_score

    def adaboost_objective(trial,X_train_data,y_train_data):
        learning_rate = trial.suggest_float('learning_rate',0.001,2)
        n_estimators = trial.suggest_int('n_estimators', 10, 100,1)
        clf = AdaBoostClassifier(learning_rate=learning_rate, n_estimators=n_estimators, random_state=42)
        matthews_corrcoef,f2_score = model_trainer.classification_metrics(clf,X_train_data,y_train_data)
        return matthews_corrcoef,f2_score

    def xgboost_objective(trial,X_train_data,y_train_data):
        booster = trial.suggest_categorical('booster',['gbtree','gblinear'])
        eta = trial.suggest_float('eta',0,0.2)
        gamma = trial.suggest_float('gamma',0,10)
        max_depth = trial.suggest_int('max_depth',1,10)
        min_child_weight = trial.suggest_float('min_child_weight',0.5,3)
        max_delta_step = trial.suggest_int('max_delta_step',0,10)
        subsample = trial.suggest_float('subsample',0,1)
        colsample_bytree = trial.suggest_float('colsample_bytree',0,1)
        colsample_bylevel = trial.suggest_float('colsample_bylevel',0,1)
        colsample_bynode = trial.suggest_float('colsample_bynode',0,1)
        lambdas = trial.suggest_float('lambda',0,2)
        alpha = trial.suggest_float('alpha',0,2)
        sample_type = trial.suggest_categorical('sample_type',['uniform','weighted'])
        normalize_type = trial.suggest_categorical('normalize_type',['tree','forest'])
        rate_drop = trial.suggest_float('rate_drop',0,1)
        one_drop = trial.suggest_float('one_drop',0,1)
        skip_drop = trial.suggest_float('skip_drop',0,1)
        clf = XGBClassifier(objective='binary:logistic', eval_metric='logloss', verbosity=0, booster=booster,
                        eta=eta, gamma=gamma, max_depth=max_depth, min_child_weight=min_child_weight, 
                        max_delta_step=max_delta_step, subsample=subsample, colsample_bytree=colsample_bytree,
                        colsample_bylevel=colsample_bylevel, colsample_bynode=colsample_bynode, lambdas=lambdas,
                        alpha=alpha, sample_type=sample_type, normalize_type=normalize_type, rate_drop=rate_drop,
                        one_drop=one_drop, skip_drop=skip_drop, random_state=42)
        matthews_corrcoef,f2_score = model_trainer.classification_metrics(clf,X_train_data,y_train_data)
        return matthews_corrcoef,f2_score
        
    def classification_metrics(clf,X_train_data,y_train_data):
        matthews_corrcoef_score = cross_val_score(clf, X_train_data, y_train_data, cv=5, scoring= make_scorer(matthews_corrcoef)).mean()
        f2_score = cross_val_score(clf, X_train_data, y_train_data, cv=5, scoring=make_scorer(fbeta_score, beta=2, average='macro')).mean()
        return matthews_corrcoef_score, f2_score

    def initialize_model_training(self,folderpath,filepath):
        self.log_writer.log(self.file_object, 'Start initializing objectives required for model training')
        self.filepath = filepath
        objectives = [model_trainer.rf_objective, model_trainer.lr_objective, model_trainer.dt_objective, 
        model_trainer.et_objective, model_trainer.svc_objective, model_trainer.knn_objective, model_trainer.gaussiannb_objective,
        model_trainer.adaboost_objective, model_trainer.gradientboost_objective, model_trainer.xgboost_objective]
        selectors = [RandomForestClassifier(random_state=42), LogisticRegression(max_iter=1000000,random_state=42), 
        DecisionTreeClassifier(random_state=42), ExtraTreesClassifier(random_state=42), 
        SVC(probability=True, kernel='linear', random_state=42), KNeighborsClassifier(), GaussianNB(), 
        AdaBoostClassifier(random_state=42), GradientBoostingClassifier(random_state=42, max_depth=None), 
        XGBClassifier(objective='binary:logistic', eval_metric='logloss', verbosity=0, random_state=42)]
        try:
            results = pd.concat([pd.Series(name='column_list'), pd.Series(name='num_features'), pd.Series(name='model_name'), 
            pd.Series(name='best_params'), pd.Series(name='resampling_indicator'), pd.Series(name='clustering_indicator'),
            pd.Series(name='train_matthews_corrcoef'), pd.Series(name='val_matthews_corrcoef'), pd.Series(name='train_f2_score'),
            pd.Series(name='val_f2_score'), pd.Series(name='train_val_matthews_corrcoef'), pd.Series(name='test_matthews_corrcoef'), 
            pd.Series(name='train_val_f2_score'),pd.Series(name='test_f2_score')], axis=1)
            results.to_csv(folderpath+filepath, mode='w',index=False)
        except Exception as e:
            self.log_writer.log(self.file_object, f'Fail to create initial csv file of results from model training with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, 'Finish initializing objectives required for model training')
        return objectives, selectors

    def resample_data(self,X,y_sub):
        self.X = X
        self.y_sub = y_sub
        self.log_writer.log(self.file_object, 'Start resampling data')
        sampler = SMOTETomek(random_state=42)
        X_res, y_res = sampler.fit_resample(self.X, self.y_sub)
        self.log_writer.log(self.file_object, 'Finish resampling data')
        return X_res, y_res

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
            study.optimize(obj, n_trials=n_trials)
            trials = study.best_trials
        except Exception as e:
            self.log_writer.log(self.file_object, f'Performing optuna hyperparameter tuning for {obj.__name__} model failed with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish performing optuna hyperparameter tuning for {obj.__name__} model')
        return trials

    def model_evaluation(self, model, trial, X_train_data, y_train_data, X_test_data, y_test_data):
        self.log_writer.log(self.file_object, f'Start evaluating {type(model).__name__} model performance for trial {trial.number}')
        try:
            model.set_params(**trial.params)
            cv_results = cross_validate(model, X_train_data, y_train_data, cv=5, return_train_score=True,
            scoring={"matthews_corrcoef": make_scorer(matthews_corrcoef), "f2_score_macro": make_scorer(fbeta_score, beta=2, average='macro')})
            if cv_results['train_matthews_corrcoef'].mean() == 0 or cv_results['test_matthews_corrcoef'].mean() == 0:
                self.log_writer.log(self.file_object, f'Trial {trial.number} was not used for model evaluation because there is no predictions made on minority class')
                return None, None, None, None, None
            model.fit(X_train_data, y_train_data)
            y_train_pred = model.predict(X_train_data)
            y_test_pred = model.predict(X_test_data)
            train_val_matthews_corrcoef = matthews_corrcoef(y_train_data, y_train_pred)
            test_matthews_corrcoef = matthews_corrcoef(y_test_data, y_test_pred)
            train_val_f2_score = fbeta_score(y_train_data, y_train_pred,average='macro', beta=2)
            test_f2_score = fbeta_score(y_test_data, y_test_pred,average='macro', beta=2)
        except Exception as e:
            self.log_writer.log(self.file_object, f'Evaluating {type(model).__name__} model performance for trial {trial.number} failed with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish evaluating {type(model).__name__} model performance for trial {trial.number}')
        return cv_results, train_val_matthews_corrcoef, test_matthews_corrcoef, train_val_f2_score, test_f2_score

    def train_per_model(self,obj, clf, trial_size,col_list, n_features,X_train,y_train,X_test,y_test,num_features, col_selected, model_name, best_params, resampling_yes_no, clustering_yes_no, train_matthews_corrcoef, val_matthews_corrcoef, train_f2_score, val_f2_score, train_val_matthews_corrcoef_score, test_matthews_corrcoef_score, train_val_f2_score, test_f2_score, resampling, clustering):
        self.log_writer.log(self.file_object, f'Start model training on {type(clf).__name__} for {n_features} features with {resampling} resampling and {clustering} clustering')
        try:
            func = lambda trial: obj(trial, X_train, y_train)
            func.__name__ = type(clf).__name__
            model = clone(clf)
            trials = self.optuna_optimizer(func,trial_size)
            for trial in trials:
                model = clone(clf)
                cv_results, train_val_matthews_corrcoef, test_matthews_corrcoef, train_val_f2, test_f2 = self.model_evaluation(model, trial, X_train, y_train, X_test, y_test)
                if cv_results is None:
                    continue
                num_features.append(n_features)
                col_selected.append(col_list)
                model_name.append(type(model).__name__)
                best_params.append(model.get_params())
                resampling_yes_no.append(resampling)
                clustering_yes_no.append(clustering)
                train_matthews_corrcoef.append(cv_results['train_matthews_corrcoef'].mean())
                val_matthews_corrcoef.append(cv_results['test_matthews_corrcoef'].mean())
                train_f2_score.append(cv_results['train_f2_score_macro'].mean())
                val_f2_score.append(cv_results['test_f2_score_macro'].mean())
                train_val_matthews_corrcoef_score.append(train_val_matthews_corrcoef)
                train_val_f2_score.append(train_val_f2)                
                test_matthews_corrcoef_score.append(test_matthews_corrcoef)
                test_f2_score.append(test_f2) 
            self.log_writer.log(self.file_object, f"{type(model).__name__} with {n_features} features added")
        except Exception as e:
            self.log_writer.log(self.file_object, f'Model training on {type(clf).__name__} for {n_features} features with {resampling} resampling and {clustering} clustering failed with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish model training on {type(clf).__name__} for {n_features} features with {resampling} resampling and {clustering} clustering')

    def store_tuning_results(self,col_selected,num_features,model_name,best_params,resampling_yes_no, clustering_yes_no, train_matthews_corrcoef,val_matthews_corrcoef,train_f2_score,val_f2_score,train_val_matthews_corrcoef_score,test_matthews_corrcoef_score,train_val_f2_score,test_f2_score,folderpath,filepath,n_features):
        self.log_writer.log(self.file_object, f'Start appending results from model training for {n_features} features')
        try:
            results = pd.concat([pd.Series(col_selected, name='column_list'), pd.Series(num_features, name='num_features'), 
                                pd.Series(model_name, name='model_name'), pd.Series(best_params, name='best_params'),
                                pd.Series(resampling_yes_no, name='resampling_indicator'), 
                                pd.Series(clustering_yes_no, name='clustering_indicator'), 
                                pd.Series(train_matthews_corrcoef, name='train_matthews_corrcoef'), 
                                pd.Series(val_matthews_corrcoef, name='val_matthews_corrcoef'), 
                                pd.Series(train_f2_score, name='train_f2_score'), pd.Series(val_f2_score, name='val_f2_score'),
                                pd.Series(train_val_matthews_corrcoef_score, name='train_val_matthews_corrcoef'),
                                pd.Series(test_matthews_corrcoef_score, name='test_matthews_corrcoef'),
                                pd.Series(train_val_f2_score, name='train_val_f2_score'),
                                pd.Series(test_f2_score, name='test_f2_score'),], axis=1)
            results.to_csv(folderpath+filepath, mode='a',header=False, index=False)
        except Exception as e:
            self.log_writer.log(self.file_object, f'Appending results from model training for {n_features} features failed with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish appending results from model training for {n_features} features')

    def best_model(self, folderpath, filepath, bestresultpath, threshold):
        self.log_writer.log(self.file_object, f'Start determining best configuration to use for saving models')
        try:
            results = pd.read_csv(folderpath+filepath).sort_values(by='test_matthews_corrcoef',ascending=False)
            best_models = results[(np.abs(results['train_val_matthews_corrcoef'] - results['test_matthews_corrcoef'])<threshold) & (np.abs(results['train_val_f2_score'] - results['test_f2_score'])<threshold)].reset_index(drop=True)
            final_models = best_models[(best_models['test_matthews_corrcoef'] == best_models['test_matthews_corrcoef'].max()) & (best_models['test_f2_score'] == best_models['test_f2_score'].max())].sort_values(by=['num_features','resampling_indicator','clustering_indicator'])
            # If no model performs best for both metrics, then pick the model with the highest matthews correlation coefficient on the test set
            if len(final_models) == 0:
                final_models = best_models[(best_models['test_f2_score'] == best_models['test_f2_score'].max())].sort_values(by=['num_features','resampling_indicator','clustering_indicator'])
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
                kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
                kmeans.fit(data)
                wcss.append(kmeans.inertia_)
            kneeloc = KneeLocator(range(start_cluster,end_cluster+1), wcss, curve='convex', direction='decreasing')
            kmeans=KMeans(n_clusters=kneeloc.knee,init='k-means++',random_state=42)
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

    def data_scaling_train_test(self, train_data, gaussian_variables, test_data, non_gaussian_variables):
        self.log_writer.log(self.file_object, f'Start performing scaling data on train and test set')
        try:
            scaler = self.fit_scaled_data(train_data[gaussian_variables], StandardScaler())
            gaussian_train_data_scaled = self.transform_scaled_data(train_data[gaussian_variables], scaler)
            gaussian_test_data_scaled = self.transform_scaled_data(test_data[gaussian_variables], scaler)
            scaler = self.fit_scaled_data(train_data[non_gaussian_variables], MinMaxScaler())
            non_gaussian_train_data_scaled = self.transform_scaled_data(train_data[non_gaussian_variables], scaler)
            non_gaussian_test_data_scaled = self.transform_scaled_data(test_data[non_gaussian_variables], scaler)
            X_train_scaled = pd.concat([gaussian_train_data_scaled, non_gaussian_train_data_scaled], axis=1)
            X_test_scaled = pd.concat([gaussian_test_data_scaled, non_gaussian_test_data_scaled], axis=1)
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
        ax1.set_ylabel("F2 score")
        ax1.set_title("Learning Curve Analysis (CV=5)")
        ax1.grid()
        ax1.annotate(np.round(train_score_m[-1],4),(train_size[-1]-50,train_score_m[-1]+0.05))
        ax1.annotate(np.round(test_score_m[-1],4),(train_size[-1]-50,test_score_m[-1]-0.05))
        plt.savefig(folderpath+'Learning_Curve_Analysis.png')

    def train_overall_model(self, X_train_sub, X_test_sub, y_train_data, y_test_data, model, final_result, name_model, folderpath):
        self.log_writer.log(self.file_object, f'Start training and saving the {name_model} model')
        try:
            X_sub = pd.concat([X_train_sub, X_test_sub]).reset_index(drop=True)
            y_sub = pd.concat([y_train_data, y_test_data]).reset_index(drop=True)
            model.set_params(**eval(final_result['best_params'].values[0].replace("\'missing\': nan,","").replace("'", "\"")))
            model.fit(X_sub,y_sub)
            ConfusionMatrixDisplay.from_predictions(y_sub, model.predict(X_sub)).plot()
            plt.title('Overall confusion matrix')
            plt.savefig(folderpath+'Overall_Confusion_Matrix.png')
            report = classification_report(y_sub, model.predict(X_sub), output_dict=True)
            f_beta_results = pd.Series([fbeta_score(y_sub, model.predict(X_sub), average=None, beta=2)[0],fbeta_score(y_sub, model.predict(X_sub), average=None, beta=2)[1],
            fbeta_score(y_sub, model.predict(X_sub), average='micro', beta=2),fbeta_score(y_sub, model.predict(X_sub), average='macro', beta=2),
            fbeta_score(y_sub, model.predict(X_sub), average='weighted', beta=2)], name='f2_score', index =['-1', '1', 'accuracy', 'macro avg', 'weighted avg'])
            pd.concat([pd.DataFrame(report).transpose(),f_beta_results],axis=1).to_csv(folderpath + name_model + '_Classification_Report.csv')
            train_size, train_score, test_score = learning_curve(estimator=model, X=X_sub, y=y_sub, cv=5, scoring=make_scorer(fbeta_score, beta=2))
            train_score_m = np.mean(np.abs(train_score), axis=1)
            test_score_m = np.mean(np.abs(test_score), axis=1)
            self.learning_curve_plot(folderpath, train_size, train_score_m, test_score_m)
        except Exception as e:
            self.log_writer.log(self.file_object, f'Training and saving the {name_model} model failed with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish training and saving the {name_model} model')
        return model

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
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        objectives, selectors = self.initialize_model_training(self.folderpath,self.filepath)
        gaussian_variables = list(pd.read_csv(self.folderpath+'Gaussian_columns.csv')['Variable'])
        non_gaussian_variables = list(pd.read_csv(self.folderpath+'Non_gaussian_columns.csv')['Variable'])
        X_train_scaled, X_test_scaled = self.data_scaling_train_test(self.train_data, gaussian_variables, self.test_data, non_gaussian_variables)
        X_train_res, y_train_res = self.resample_data(self.train_data, self.train_output)
        X_train_res_scaled, X_test_res_scaled = self.data_scaling_train_test(X_train_res, gaussian_variables, self.test_data, non_gaussian_variables)

        for n_features in range(1,min(21,len(X_train_scaled.columns)+1)):
            num_features, col_selected, model_name, best_params, resampling_yes_no, clustering_yes_no  = [], [], [], [], [], []
            train_matthews_corrcoef, val_matthews_corrcoef, train_f2_score, val_f2_score = [], [], [], []
            train_val_matthews_corrcoef_score, test_matthews_corrcoef_score, train_val_f2_score, test_f2_score = [], [], [], []
            for obj, clf in zip(objectives, selectors):
                for resampling in ['yes','no']:
                    if resampling == 'yes':
                        X_train_data, y_train_data = self.scale_vs_non_scale_data(clf,X_train_res_scaled,X_train_res,y_train_res)
                        X_test_data, y_test_data = self.scale_vs_non_scale_data(clf,X_test_res_scaled,self.test_data,self.test_output)
                    else:
                        X_train_data, y_train_data = self.scale_vs_non_scale_data(clf,X_train_scaled,self.train_data,self.train_output)
                        X_test_data, y_test_data = self.scale_vs_non_scale_data(clf,X_test_scaled,self.test_data,self.test_output)                       
                    transformer = SelectKBest(f_classif, k=n_features)
                    X_train_sub = pd.DataFrame(transformer.fit_transform(X_train_data, y_train_data))
                    X_test_sub= pd.DataFrame(transformer.transform(X_test_data))
                    col_list = X_train_data.columns[transformer.get_support()].tolist()
                    X_train_sub.columns = col_list
                    X_test_sub.columns = col_list
                    for clustering in ['yes', 'no']:
                        col_list = X_train_data.columns[transformer.get_support()].tolist()
                        if clustering == 'yes':
                            X_train_scaled_sub = pd.DataFrame(transformer.transform(X_train_res_scaled), columns = col_list) if resampling == 'yes' else pd.DataFrame(transformer.transform(X_train_scaled), columns = col_list)
                            X_test_scaled_sub = pd.DataFrame(transformer.transform(X_test_scaled), columns = col_list)
                            X_train_cluster_sub, X_test_cluster_sub, X_train_scaled_cluster_sub, X_test_scaled_cluster_sub = self.add_cluster_number_to_data(X_train_scaled_sub, X_train_sub, X_test_scaled_sub, X_test_sub)
                            X_train_cluster_data, y_train_data = self.scale_vs_non_scale_data(clf,X_train_scaled_cluster_sub,X_train_cluster_sub,y_train_data)
                            X_test_cluster_data, y_test_data = self.scale_vs_non_scale_data(clf,X_test_scaled_cluster_sub,X_test_cluster_sub,y_test_data) 
                            col_list.extend(pd.get_dummies(X_train_cluster_data['cluster'], drop_first=True).columns.tolist())
                            X_train_cluster_data = pd.concat([X_train_cluster_data.iloc[:,:-1], pd.get_dummies(X_train_cluster_data['cluster'], drop_first=True)],axis=1)
                            X_test_cluster_data = pd.concat([X_test_cluster_data.iloc[:,:-1], pd.get_dummies(X_test_cluster_data['cluster'], drop_first=True)],axis=1)
                            self.train_per_model(obj, clf, 20, col_list, n_features,X_train_cluster_data,y_train_data,X_test_cluster_data,
                            y_test_data,num_features, col_selected, model_name, best_params, resampling_yes_no, clustering_yes_no,
                            train_matthews_corrcoef, val_matthews_corrcoef, train_f2_score, val_f2_score, train_val_matthews_corrcoef_score, 
                            test_matthews_corrcoef_score, train_val_f2_score, test_f2_score, resampling, clustering)
                        else:
                            self.train_per_model(obj, clf, 20, col_list, n_features,X_train_sub,y_train_data,X_test_sub,
                            y_test_data,num_features, col_selected, model_name, best_params, resampling_yes_no, clustering_yes_no,
                            train_matthews_corrcoef, val_matthews_corrcoef, train_f2_score, val_f2_score, train_val_matthews_corrcoef_score, 
                            test_matthews_corrcoef_score, train_val_f2_score, test_f2_score, resampling, clustering)                            
            self.store_tuning_results(col_selected,num_features,model_name,best_params, resampling_yes_no, clustering_yes_no, 
            train_matthews_corrcoef,val_matthews_corrcoef,train_f2_score, val_f2_score, train_val_matthews_corrcoef_score,test_matthews_corrcoef_score,
            train_val_f2_score, test_f2_score, self.folderpath, self.filepath,n_features)
        
        final_result = self.best_model(self.folderpath, self.filepath, self.bestresultpath, self.threshold)
        name_model = final_result['model_name'].values[0]
        num_features = final_result['num_features'].values[0]
        resampling = final_result['resampling_indicator'].values[0]
        clustering = final_result['clustering_indicator'].values[0]
        model_dict = {'RandomForestClassifier':RandomForestClassifier(),'LogisticRegression':LogisticRegression(),
        'DecisionTreeClassifier':DecisionTreeClassifier(),'SVC':SVC(),'KNeighborsClassifier':KNeighborsClassifier(),
        'GaussianNB':GaussianNB(),'XGBClassifier':XGBClassifier(),'ExtraTreesClassifier':ExtraTreesClassifier(),
        'AdaBoostClassifier':AdaBoostClassifier(),'GradientBoostingClassifier':GradientBoostingClassifier()}
        model = model_dict[name_model]
        X_train_data, y_train_data = self.scale_vs_non_scale_data(model,X_train_res_scaled,X_train_res,y_train_res) if resampling == 'yes' else self.scale_vs_non_scale_data(model,X_train_scaled,self.train_data,self.train_output)
        X_test_data, y_test_data = self.scale_vs_non_scale_data(model,X_test_scaled,self.test_data,self.test_output)
        columns = final_result['column_list'].values[0].replace("'","").strip("][").split(', ')[:num_features]
        X_train_sub = X_train_data[columns]
        X_test_sub = X_test_data[columns]

        if clustering == 'yes':
            X_train_scaled_sub = X_train_res_scaled[columns] if resampling == 'yes' else X_train_scaled[columns]
            X_test_scaled_sub = X_test_scaled[columns]
            X_train_cluster_sub, X_test_cluster_sub, X_train_scaled_cluster_sub, X_test_scaled_cluster_sub = self.add_cluster_number_to_data(X_train_scaled_sub, X_train_sub, X_test_scaled_sub, X_test_sub, final=True)
            X_train_cluster_data, y_train_data = self.scale_vs_non_scale_data(model,X_train_scaled_cluster_sub,X_train_cluster_sub,y_train_data)
            X_test_cluster_data, y_test_data = self.scale_vs_non_scale_data(model,X_test_scaled_cluster_sub,X_test_cluster_sub,y_test_data) 
            X_train_sub = pd.concat([X_train_cluster_data.iloc[:,:-1], pd.get_dummies(X_train_cluster_data['cluster'], drop_first=True)],axis=1)
            X_test_sub = pd.concat([X_test_cluster_data.iloc[:,:-1], pd.get_dummies(X_test_cluster_data['cluster'], drop_first=True)],axis=1)
        trained_model = self.train_overall_model(X_train_sub, X_test_sub, y_train_data, y_test_data, model, final_result, name_model, self.folderpath)
        pkl.dump(trained_model,open('Saved_Models/'+name_model+'_'+resampling+'_resampling_'+clustering+'_clustering.pkl','wb'))
        self.log_writer.log(self.file_object, 'Finish model training and hyperparameter tuning')