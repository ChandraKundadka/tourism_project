# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# For dispplaying debug information
from huggingface_hub.utils import logging
logging.set_verbosity_debug()

#For experiment tracking
import mlflow

 #For Enable debug logging
from huggingface_hub.utils import logging

print("=" * 70)
print("Starting train.py".center(70))
print("=" * 70)

logging.set_verbosity_debug()

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

#api = HfApi()
#print(api)
api = HfApi(token=os.getenv("HF_TOURISM"))

# Define repo_id and repo_type explicitly in this script
repo_id = "Chandrashekhara/tourism-project"
repo_type = "dataset"                        # This is important for the hf://datasets/ prefix

#Listing the files to check the files which are available on the dataspace.
print("List the files ")
print(' ')
files = api.list_repo_files(
    repo_id=repo_id,
    repo_type=repo_type,
)

print(files)
print(' ')

Xtrain_path= "hf://datasets/Chandrashekhara/tourism-project/Xtrain.csv"
Xtest_path= "hf://datasets/Chandrashekhara/tourism-project/Xtest.csv"
ytrain_path= "hf://datasets/Chandrashekhara/tourism-project/ytrain.csv"
ytest_path= "hf://datasets/Chandrashekhara/tourism-project/ytest.csv"


Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# List of numerical features in the dataset
numeric_features = [
"Age",	                    #Age of the customer.
"NumberOfPersonVisiting",	  #Total number of people accompanying the customer on the trip.
"PreferredPropertyStar",	  #Preferred hotel rating by the customer.
"NumberOfTrips",	          #Average number of trips the customer takes annually.
"NumberOfChildrenVisiting",	#Number of children below age 5 accompanying the customer.
"MonthlyIncome",	          #Gross monthly income of the customer.
"PitchSatisfactionScore",	  #Score indicating the customer's satisfaction with the sales pitch.
"NumberOfFollowups",	      #Total number of follow-ups by the salesperson after the sales pitch.
"DurationOfPitch" 	        #Duration of the sales pitch delivered to the customer.
]

# List of categorical features in the dataset
categorical_features = [
"TypeofContact",	  #The method by which the customer was contacted (Company Invited or Self Inquiry).
"Occupation",   		#Customer's occupation (e.g., Salaried, Freelancer).
"Gender",		        #Gender of the customer (Male, Female).
"ProductPitched",		#The type of product pitched to the customer.
"MaritalStatus",		#Marital status of the customer (Single, Married, Divorced).
"Designation", 		  #Customer's designation in their current organization.
"CityTier",	                #The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3).
"Passport",	                #Whether the customer holds a valid passport (0: No, 1: Yes).
"OwnCar" 	                  #Whether the customer owns a car (0: No, 1: Yes).
]

print(f"numeric_features: {numeric_features}")
print(f"categorical_features: {categorical_features}")

# Class weight to handle imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessing pipeline
# One-hot encode categorical and scale numeric features
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}


# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, scoring='recall', cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "Chandrashekhara/tourism-project-model"
    repo_type = "model"

    token = os.getenv("HF_TOURISM")
    if token is None:
        raise ValueError("HF_TOURISM is not set!")

    print("Token found:", token[:8] + "...")

    #api = HfApi(token=os.getenv("HF_TOURISM"))
    api = HfApi(token=token)

    # Step 1: Check if the space exists
    try:
      api.repo_info(repo_id=repo_id, repo_type=repo_type)
      print(f"Repository '{repo_id}' already exists.")
    except RepositoryNotFoundError:
       # Step 2: Create if the space not exists
      print(f"Going to Create Repository '{repo_id} ' repo_type=' {repo_type}")
      try:
          create_repo(
              repo_id=repo_id,
              repo_type=repo_type,
              token=token,
              private=False
          )
          print("Repository created successfully.")
      except Exception as e:
          print(f"Error creating repository: {e}")

    api.upload_file(
        path_or_fileobj="best_tourism_model_v1.joblib",
        path_in_repo="best_tourism_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Model uploaded successfully to '{repo_id}'!")

print("=" * 70)
print("Finishing train.py".center(70))
print("=" * 70)
