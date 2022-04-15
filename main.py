from Model_Prediction_Modules.validation_pred_data import rawpreddatavalidation
from Model_Prediction_Modules.pred_preprocessing import pred_Preprocessor
from Model_Prediction_Modules.model_prediction import model_predictor
from Model_Training_Modules.validation_train_data import rawtraindatavalidation
from Model_Training_Modules.train_preprocessing import train_Preprocessor
from Model_Training_Modules.model_training import model_trainer
import streamlit as st
import os
from zipfile import ZipFile

def main():
    st.title("Wafer Status Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Wafer Status Prediction App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    if st.button("Model Training"):
        with open("Training_Logs/Training_Main_Log.txt", 'a+') as file:
            trainvalidator = rawtraindatavalidation('sensordata', file, "Good_Training_Data/", "Bad_Training_Data/")
            folders = ['Good_Training_Data/','Bad_Training_Data/','Archive_Training_Data/','Training_Data_FromDB/','Intermediate_Train_Results/']
            trainvalidator.initial_data_preparation('schema_training.json',folders,"Training_Batch_Files",'Good_Training_Data/','Archive_Training_Data/','Training_Data_FromDB/Training_Data.csv') 
        with open("Training_Logs/Training_Preprocessing_Log.txt", 'a+') as file:
            preprocessor = train_Preprocessor(file, 'Intermediate_Train_Results/')
            X_train, X_test, y_train, y_test = preprocessor.data_preprocessing('Training_Data_FromDB/Training_Data.csv', 'Columns_Drop_from_Original.csv','Wafer','Output')
        with open("Training_Logs/Training_Model_Log.txt", 'a+') as file:
            trainer = model_trainer(file)
            trainer.train_model_and_hyperparameter_tuning(X_train, X_test, y_train, y_test,'Intermediate_Train_Results/','Model_results_by_num_features.csv','Best_Model_Results.csv', 0.05)
        with ZipFile("training_files_download.zip", "w") as newzip:
            for folder in ['Archive_Training_Data','Training_Data_FromDB','Intermediate_Train_Results','Training_Logs','Saved_Models']:
                for file in os.listdir(folder):
                    newzip.write(os.path.join(folder,file))
        with open("training_files_download.zip", "rb") as zipfile:
            st.download_button(label="Download 7z",data=zipfile,file_name="download.zip",mime="application/zip")
        st.success('Model training process is complete')

    if st.button("Data Prediction"):
        with open("Prediction_Logs/Prediction_Main_Log.txt", 'a+') as file:
            predvalidator = rawpreddatavalidation('predsensordata', file, "Good_Prediction_Data/", "Bad_Prediction_Data/")
            folders = ['Good_Prediction_Data/','Bad_Prediction_Data/','Archive_Prediction_Data/','Prediction_Data_FromDB/','Intermediate_Pred_Results/']
            predvalidator.initial_data_preparation('schema_prediction.json',folders,"Prediction_Batch_Files/",'Good_Prediction_Data/','Archive_Prediction_Data/','Prediction_Data_FromDB/Prediction_Data.csv')
        with open("Prediction_Logs/Prediction_Preprocessing_Log.txt", 'a+') as file:
            preprocessor = pred_Preprocessor(file)
            X_pred = preprocessor.data_preprocessing('Prediction_Data_FromDB/Prediction_Data.csv','Intermediate_Train_Results/Best_Model_Results.csv')
        with open("Prediction_Logs/Prediction_Model_Log.txt", 'a+') as file:
            model_objects = os.listdir('Saved_Models/')
            if len(model_objects)>1:
                model_objects.remove('kmeans_model.pkl')    
            predictor = model_predictor(file)
            predictor.model_prediction('Intermediate_Train_Results/Best_Model_Results.csv',f'Saved_Models/{model_objects[0]}', 'Saved_Models/kmeans_model.pkl', X_pred,'Wafer')
        with ZipFile("prediction_files_download.zip", "w") as newzip:
            for folder in ['Archive_Prediction_Data','Prediction_Data_FromDB','Intermediate_Pred_Results','Prediction_Logs']:
                for file in os.listdir(folder):
                    newzip.write(os.path.join(folder,file))
        with open("prediction_files_download.zip", "rb") as zipfile:
            st.download_button(label="Download 7z",data=zipfile,file_name="download.zip",mime="application/zip")
        st.success('Data prediction process is complete')
        
if __name__=='__main__':
    main()