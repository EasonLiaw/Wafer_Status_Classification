# Wafer Status Classification Project

**Background**
---

<img src="https://www.semiconductorforu.com/wp-content/uploads/2021/02/silicon-wafer.jpg">

In electronics, a wafer (also called a slice or substrate) is a thin slice of semiconductor used for the fabrication of integrated circuits. Monitoring working conditions of these wafers present its challenges of having additional resources required for manual monitoring with insights and decisions that need to be made quickly for replacing wafers that are not in good working conndition when required. Using IIOT (Industrial Internet of Things) helps to overcome this challenge through a collection of real-time data from multiple sensors. 

Thus, the main goal of this project is to design a machine learning model that predicts whether a wafer is in a good working condition or not based on inputs from 590 different sensors for every wafer. The quality of wafer sensors can be classified into two different categories: -1 for "good wafer" and +1 for "bad wafer".

Dataset is provided in .csv format by client under <b>Training_Batch_Files</b> folder for model training, while dataset under <b>Prediction_Batch_Files</b> folder will be used for predicting quality of wafer sensors.

In addition, schema of datasets for training and prediction is provided in .json format by the client for storing seperate csv files into a single MySQL database.

**Code and Resources Used**
---
- **Python Version** : 3.9.10
- **Packages** : feature-engine, imbalanced-learn, kneed, matplotlib, mysql-connector-python, numpy, optuna, pandas, scikit-learn, scipy, seaborn, streamlit
- **Dataset source** : Education materials from OneNeuron platform
- **Streamlit documentation** : https://spotipy.readthedocs.io/en/2.16.0/
- **Optuna documentation** : https://spotipy.readthedocs.io/en/2.16.0/

**Project Architecture Summary**
---
The following points below summarizes the structure for running this project:
1. Validating the quality of batch files source (<b>Training_Batch_Files</b> folder) based on a set of customized validation rules for model training.
2. Inserting good quality training data from batch files into MySQL database, which is hosted on Heroku platform.
3. Export compiled data from previous step into a single .csv file in <b>Training_Data_FromDB</b> folder for data preprocessing.
4. Preprocess data using various feature engineering techniques, while performing initial Exploratory Data Analysis (EDA). Note that results from certain stages of data preprocessing are saved in <b>Intermediate_Train_Results</b> folder for reference.
5. Perform resampling using <b>SMOTETomek</b> on the training dataset seperately after data preprocessing.
6. The following set of configurations are also tested for various number of features selected from using <b>F-classification</b> on various models along with hyperparameter tuning:
- Resampling vs No resampling
- Clustering vs No clustering
7. Results from previous steps are stored in .csv file named as <b>"Model_results_by_num_features.csv"</b>
8. Best model is identified from a set of evaluation criteria and the best model is retrained on the entire dataset before saving it into .pkl file under <b>Saved_Models</b> folder.
9. Validating the quality of batch files source (<b>Prediction_Batch_Files</b> folder) based on a set of customized validation rules for model prediction.
10. Inserting good quality prediction data from batch files into MySQL database.
11. Export compiled data from previous step into a single .csv file in <b>Prediction_Data_FromDB</b> folder for data preprocessing.
12. Preprocess data using the same feature engineering techniques identified in Step 4.
13. Perform either resampling or clustering or both on the preprocessed data from previous step based on configurations of best model identified from step 8.
14. Perform model prediction using best models saved from step 8 and results from model prediction are saved into <b>Intermediate_Pred_Results</b> folder.

Note that all steps mentioned above have been logged accordingly for future reference and easy maintenance, which are stored in <b>Training_Logs</b> and <b>Prediction_Logs</b> folders. Any bad quality data identified for model training and model prediction will be archived accordingly in <b>Archive_Training_Data</b> and <b>Archive_Prediction_Data</b> folders.

The following sections below explains the three main approaches that can be used for model deployment in this project:
1. <b>Cloud Platform (Heroku)</b>
2. <b>Local system</b>
3. <b>Docker</b>

**Project Instructions (Heroku)**
---
<b> For deploying models onto Heroku platform, the following additional files are essential</b>:
- Procfile
- requirements.txt
- setup.sh

<b>Note that deploying models onto other cloud platforms like GCP, AWS or Azure may have different additionnal files required.</b>

For replicating the steps required for running this project on your own Heroku account, the following steps are required:
1. Clone this github repository into your local machine system or your own Github account if available.
<img src="https://user-images.githubusercontent.com/34255556/160223555-388eb56f-7ca5-4d40-88ca-ad80b1638599.png" width="600" height="200">

2. Delete files stored inside Training_Logs and Prediction_Logs folder, while creating a dummy.txt file inside both folders respectively. This is to ensure both directories exist when the model is deployed into Heroku.
<img src="https://user-images.githubusercontent.com/34255556/160224012-4f861309-1e7a-40ad-b466-dbdc8e22f20e.png" width="600" height="80">

3. Go to your own Heroku account and create a new app with your own customized name.
<img src="https://user-images.githubusercontent.com/34255556/160223589-301262f6-6225-4962-a92f-fc7ca8a0eee9.png" width="600" height="400">

4. Go to "Resources" tab and search for ClearDB MySQL in the add-ons search bar.
<img src="https://user-images.githubusercontent.com/34255556/160224064-35295bf6-3170-447a-8eae-47c6721cf8f0.png" width="600" height="200">

5. Select the ClearDB MySQL add-on and select the relevant pricing plan. (Note that I select Punch plan, which currently cost about $9.99 per month to increase storage capacity for this project.)

6. Add an additional Python file named as DBConnectionSetup.py that contains the following Python code structure: logins = {"host": <host_name>, "user": <user_name>, "password": <password>, "dbname": <default_Heroku_database_name>}. 
- For security reasons, this file needs to be stored in private. I've also included a video reference link below for clear instructions on setup ClearDB MySQL for Heroku.
  
[![Deploy MySQL Database into Heroku](https://i.ytimg.com/vi/Zcg71lxW-Yo/hqdefault.jpg)](https://www.youtube.com/watch?v=Zcg71lxW-Yo&ab_channel=CodeJava)

7. Inside your new app, deploy the code into the app by either linking your github repository or manually deploy it using Heroku CLI (Instructions are available and self-explanatory when selecting Heroku CLI option).
<img src="https://user-images.githubusercontent.com/34255556/160223941-2aacc3ca-4ab5-4996-be46-f2d553933dd5.png" width="600" height="300">

8. After successful model deployment, open the app and you will see the following interface designed using Streamlit:
<img src="https://user-images.githubusercontent.com/34255556/160223634-131bcd65-9c37-47dd-b71c-0de15b6b93ed.png" width="600" height="250">

9. From the image above, click on Model_Training first for training models, followed by Data_Prediction for model prediction. (Note that zip files will be available for download after successfully completing either model training or model prediction, which contains the results from running the process)
<img src="https://user-images.githubusercontent.com/34255556/160225941-867f92e0-5709-44f5-b35e-426187c5a99b.png" width="600" height="400">
  
<b>Important Note: Using "free" dynos on Heroku app only allows the app to run for a maximum of 30 minutes. Since the model training and prediction process takes a long time, consider changing the dynos type to "hobby" for unlimited time, which cost about $7 per month per dyno. You may also consider changing the dynos type to Standard 1X/2X for enhanced app performance.</b>

**Project Instructions (Local Environment)**
---  
If you prefer to deploy this project on your local machine system, the steps for deploying this project has been simplified down to the following:

1. Download and extract the zip file from this github repository into your local machine system.
<img src="https://user-images.githubusercontent.com/34255556/160223555-388eb56f-7ca5-4d40-88ca-ad80b1638599.png" width="600" height="200">

2. Empty the files stored inside Training_Logs and Prediction_Logs folder.

3. Open MySQL in your local machine system and create a new database name of your choice with the following syntax: <b>CREATE DATABASE db_name;</b>
- Note that you will need to install MySQL if not available in your local system: https://dev.mysql.com/downloads/windows/installer/8.0.html
  
4. Add an additional Python file named as DBConnectionSetup.py that contains the following Python code structure: logins = {"host": <host_name>, "user": <user_name>, "password": <password>, "dbname": <new_local_database_name>}. 
- For security reasons, this file needs to be stored in private.
  
5. Open anaconda prompt and create a new environment with the following syntax: <b>conda create -n myenv python=3.9.10 anaconda</b>
- Note that you will need to install anaconda if not available in your local system: https://www.anaconda.com/

6. After creating a new anaconda environment, activate the environment using the following command: <b>conda activate myenv</b>

7. Go to the local directory where your downloaded file is located and run the following command to install all the python libraries : <b>pip install -r requirements.txt</b>

8. After installing all the required Python libraries, run the following command on your project directory: <b>streamlit run main.py</b>

If you encounter the TomlDecodeError, ensure that the <b>config.toml</b> file is removed from the directory where Streamlit is installed to prevent TomlDecodeError. The following link explains more details about the error that you might encounter: https://stackoverflow.com/questions/59811357/how-to-solve-toml-decoder-tomldecodeerror-key-group-not-on-a-line-by-itself-l

9. A new browser will open after successfully running the streamlit app with the following interface:
<img src="https://user-images.githubusercontent.com/34255556/160223634-131bcd65-9c37-47dd-b71c-0de15b6b93ed.png" width="600" height="250">

10. From the image above, click on Model_Training first for training models, followed by Data_Prediction for model prediction. (Note that zip files will be available for download after successfully completing either model training or model prediction, which contains the results from running the process)
<img src="https://user-images.githubusercontent.com/34255556/160225941-867f92e0-5709-44f5-b35e-426187c5a99b.png" width="600" height="400">

**Project Instructions (Docker)**
---
A suitable alternative for deploying this project is to use Docker, which allows easy deployment on other running instances. 
  
<b>Note that docker image is created under Windows Operating system for this project, therefore these instructions will only work on other windows instances.</b>

Docker Desktop needs to be installed into your local system, before proceeding with the following steps:

1. Download and extract the zip file from this github repository into your local machine system.
<img src="https://user-images.githubusercontent.com/34255556/160223555-388eb56f-7ca5-4d40-88ca-ad80b1638599.png" width="600" height="200">

2. Empty the files stored inside Training_Logs and Prediction_Logs folder.  
  
3. Add an additional Python file named as DBConnectionSetup.py that contains the following Python code structure: logins = {"host": <host_name>, "user": <user_name>, "password": <password>, "dbname": <default_database_name>}. 
- For security reasons, this file needs to be stored in private.
  
4. Create a file named Dockerfile with the following commands:
<img src="https://user-images.githubusercontent.com/34255556/160229685-c268b253-02f2-42f3-912a-189930a997f4.png">

5. Build a new docker image on the project directory with the following command: <b>docker build -t api-name .</b>

6. Run the docker image on the project directory with the following command: <b>docker run -p 8501:8501 api-name</b>
<img src="https://user-images.githubusercontent.com/34255556/160229611-1e20ef06-dba2-4b0c-8735-2ac44fc1d38f.png" width="600" height="100">

- Note that port 8501 is required to run streamlit on Docker.

7. Open a new browser with the following link shown on the image above, which will open after successfully running the streamlit app with the following interface:
<img src="https://user-images.githubusercontent.com/34255556/160223634-131bcd65-9c37-47dd-b71c-0de15b6b93ed.png" width="600" height="250">

8. From the image above, click on Model_Training first for training models, followed by Data_Prediction for model prediction. (Note that zip files will be available for download after successfully completing either model training or model prediction, which contains the results from running the process)
<img src="https://user-images.githubusercontent.com/34255556/160225941-867f92e0-5709-44f5-b35e-426187c5a99b.png" width="600" height="400">

**Project Findings**
---

**Legality**
---
This is a personal project made for non-commercial uses ONLY. This project will not be used to generate any promotional or monetary value for me, the creator, or the user.
