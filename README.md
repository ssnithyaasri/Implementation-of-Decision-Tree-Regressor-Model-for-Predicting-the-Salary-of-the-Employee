# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. calculate Mean square error,data prediction and r2.

## Program:

/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by:NITHYAA SRI S S 
RegisterNumber:212222230100  
*/
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
import matplotlib.pyplot as plt


plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()
```

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)
![Screenshot 2024-04-02 204148](https://github.com/ssnithyaasri/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119122478/4a7425d0-fb28-47c3-97f9-239c99279db0)
![Screenshot 2024-04-02 204218](https://github.com/ssnithyaasri/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119122478/afb4be72-4a52-494b-b12a-c71d68a5a22d)
![Screenshot 2024-04-02 204230](https://github.com/ssnithyaasri/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119122478/7c9e726b-a462-480a-9453-9cb5828af3c9)
![Screenshot 2024-04-02 204249](https://github.com/ssnithyaasri/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119122478/6e3418ad-f69e-489d-9738-fe2a1c4b0db6)






## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
