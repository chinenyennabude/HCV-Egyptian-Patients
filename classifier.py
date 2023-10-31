#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# load dataset
missing_values = ['na','--','n/a',' '] #creating a list of empty values
df = pd.read_csv('HCV-Egy-Data.csv', na_values = missing_values)

#feature selection
hcv_data = df[['Age ', 'Gender', 'BMI', 'Fever','Nausea/Vomting','Headache ','Diarrhea ','Fatigue & generalized bone ache ','Jaundice ','Epigastric pain ']]


#adding a new column
category = pd.cut(hcv_data['Age '] , bins=[18, 30, 45, 65, 85], labels=['Youth', 'Adult', 'Elderly', 'Old'])
hcv_data.insert(5,'Age Group', category)
column_names = ['Age ', 'Gender', 'BMI', 'Fever','Nausea/Vomting','Headache ','Diarrhea ','Fatigue & generalized bone ache ','Jaundice ','Epigastric pain ','Age Group']
hcv_data = hcv_data.reindex(columns=column_names)

# define x and y variables
X = hcv_data.iloc[:, :-1]
y = hcv_data['Age Group']

#split dataset into train and test data
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state = 0)

#define and train model
classifier = LogisticRegression()
classifier.fit(x_train, y_train)


#test the model
prediction = classifier.predict(x_test)
print(prediction)

print(y_test)


# check recall, precision, f1-score
# print(classification_report(y_test,prediction))
# print(accuracy_score(y_test,prediction))