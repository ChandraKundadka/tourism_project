# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

#For Enable debug logging (Only required while debugging)
#from huggingface_hub.utils import logging
#logging.set_verbosity_debug()

#Repository id
repo_id = "Chandrashekhara/tourism-project"
repo_type = "dataset"  #Repository Type

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOURISM"))

DATASET_PATH = "hf://datasets/Chandrashekhara/tourism-project/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier
df.drop(columns=['CustomerID'], inplace=True)

#Gender column contains "Fe Male". Replace the value with "Female"
df["Gender"] = df["Gender"].replace("Fe Male", "Female")

#MaritalStatus contains values Unmarried and Single. Lets Change all Unmarried to Single
df["MaritalStatus"] = df["MaritalStatus"].replace("Unmarried", "Single")

# checking for duplicate values
print(f"Duplicate values:\n{df.duplicated().sum()}")

#These columns are categorical in nature but they defined as integer. Change the type to catorical.
categorical_columns = ["CityTier", "Passport", "ProdTaken","OwnCar"]
df[categorical_columns] = df[categorical_columns].astype("category")

target_col = 'ProdTaken'

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
"Designation" 		  #Customer's designation in their current organization.
"CityTier",	                #The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3).
"Passport",	                #Whether the customer holds a valid passport (0: No, 1: Yes).
"OwnCar" 	                  #Whether the customer owns a car (0: No, 1: Yes).
]


# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode categorical variables
categorical_columns = X.select_dtypes(include=['object','category']).columns
label_encoders = {}

for col in categorical_columns:
    label_encoder = LabelEncoder()
    X[col] = label_encoder.fit_transform(X[col].astype(str))
    label_encoders[col] = label_encoder


# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

#Upload resulting training and test dataset back to Hugging Face dataspace
files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id=repo_id,
        repo_type=repo_type,
    )


#List the files just to ensure files are uploaded and visible for listing.
print("List the files==>")
print(' ')
files = api.list_repo_files(
    repo_id=repo_id,
    repo_type=repo_type,
)

print(files)
