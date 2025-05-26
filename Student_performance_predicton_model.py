#------Step 1 import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#------Step 2 load the dataset and extract independent and dependent variables
student_performance = pd.read_csv('C:/Users/lenyo/Documents/Datasets/enhanced_student_habits_performance_dataset/enhanced_student_habits_performance_dataset.csv')
x = student_performance.iloc[:,:-1].values
y = student_performance.iloc[:, -1].values

#------Step 3 Data visualization(had to leave out the data that does not contain numeric values)
print(student_performance.head())
sns.heatmap(student_performance.select_dtypes(include=[np.number]).corr(), annot = True)
plt.show()

#------Step 4 Encoding the non-numeric data into numbers so that it can be understood by the computer
labelencoder = LabelEncoder()
categorical_indices = [2,3,7,10,12,13,15,19,22,23,24,28]

for i in categorical_indices:
    x[:, i] = labelencoder.fit_transform(x[:, i])

column_transformer = ColumnTransformer(transformers=[('encoder',OneHotEncoder(), categorical_indices)], remainder = 'passthrough')
x = column_transformer.fit_transform(x)

#------Step 5 added this line below to avoid dummy variable Trap
x = x[:,1:]

#------Step 6 Split the dataset into Training and Testing data
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2 , random_state = 0)

#------Step 7 Fitted the Multiple linear regression model onto the training data
model = LinearRegression()
model.fit(x_train, y_train)

#------Step 8 Use the trained model to predict the test set results
y_pred = model.predict(x_test)
print(y_pred)

#------Step 9 Calculating the coeffiecients and intercepts
print(model.coef_)
print(model.intercept_)

#------Step 10 Evaluating the model
print(r2_score(y_test,y_pred))

#------Step 11 Making the Scatter plot
sns.scatterplot(x = y_test,y = y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Performance')
plt.show()