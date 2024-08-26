# importing libs
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# loading dataset
df = pd.read_csv("/content/drive/MyDrive/Churn_Modelling.csv")
data = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# function for converting gender to the binary(0 or 1)
def gender2bin(sex):
  return 1 if "Female" else 0

# function for converting user's geography to the int (0 - Spain, 1 - France or 2 - Germany) 
def geo2int(geo):
  country_int = {"Spain": 0, "France": 1, "Germany": 2}
  return country_int[geo]

# preprocessing data
scaler = PowerTransformer()
indexes = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
data[indexes] = scaler.fit_transform(data[indexes])
data["Gender"] = data["Gender"].apply(gender2bin)
data["Geography"] = data["Geography"].apply(geo2int)

# spliting data
X, y = data[["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary", "Geography", "Gender", "HasCrCard", "IsActiveMember", "EstimatedSalary"]], data["Exited"]
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), train_size=0.8)

# using Random Forest for classification
model = RandomForestClassifier(max_depth=8, random_state=0)
model.fit(X_train, y_train) # training

# testing with the data for testing
res = model.predict(X_test)
acc = accuracy_score(y_test, res) # using accuracy metric
print(f"Model's accuracy: {acc}") # accuracy - 86.4%
