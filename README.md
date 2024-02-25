# Implementation of Simple Linear Regression Model for Predicting the Marks Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and read the dataframe.

2. Assign hours to X and scores to Y.

3. Implement training set and test set of the dataframe

4. Plot the required graph both for test data and training data.

5. Find the values of MSE , MAE and RMSE.
 

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: J JENISHA
RegisterNumber:  212222230056

```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')

df.head()
df.tail()
print(df)

X=df.iloc[:,:-1].values
print("X values = \n",X)

Y=df.iloc[:,1].values
print("Y values =\n",Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

print("Predicted values:\n",Y_pred)
print("Testing values:\n",Y_test)

plt.scatter(X_train,Y_train,color="black")
plt.plot(X_train,regressor.predict(X_train),color="purple")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,regressor.predict(X_test),color="black")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE  = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE  = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
#### Data Set
![Screenshot 2024-02-25 093034](https://github.com/Jenishajustin/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405070/d290a82e-502d-496c-96e8-6b8318e63a03)

#### Values of X
![Screenshot 2024-02-25 093149](https://github.com/Jenishajustin/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405070/6a79284e-ac3d-441b-ba05-9caf641d7287)

#### Values of Y
![Screenshot 2024-02-25 093228](https://github.com/Jenishajustin/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405070/eeb16b86-a5e8-4a83-9ceb-ae47d593140a)

#### Predicted values
![Screenshot 2024-02-25 093300](https://github.com/Jenishajustin/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405070/2369b26a-66bb-4a14-95ad-b5881066322f)

![Screenshot 2024-02-25 093338](https://github.com/Jenishajustin/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405070/627f6ae1-6285-4bd1-9998-3496e232d90a)

![Screenshot 2024-02-25 093401](https://github.com/Jenishajustin/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405070/f1959b3b-14e1-4ca3-8950-9d5ea26237b6)

#### Errors
![Screenshot 2024-02-25 093422](https://github.com/Jenishajustin/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405070/632fd059-c12c-4334-bb61-ead830563d14)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
