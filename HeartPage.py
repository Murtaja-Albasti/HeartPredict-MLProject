import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = r'C:\Users\omen\PycharmProjects\Heart_Predict_ML\heart_disease_data.csv'
df = pd.read_csv(data)
df.dropna(axis=0)
print(df['target'].value_counts())
# splitting the data
x = df.drop(columns='target', axis=1)
y = df['target']
# splitting the training testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2, stratify=y)

# build my model
# LogisticRegression model
LogisticRegression_model = LogisticRegression(random_state=2)
LogisticRegression_model.fit(x_train, y_train)

# accurcy_score validation on both models
x_predicted_Logistic = LogisticRegression_model.predict(x_test)
accurcy = accuracy_score(x_predicted_Logistic, y_test)

# build a predictive system

input_data = (57,1,0,140,192,0,1,148,0,0.4,1,0,1)
# change the input to numpy array
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = LogisticRegression_model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print('the user has not heart Disease')
else:
    print('the user has heart Disease')


output = pd.DataFrame({'Real state': y_train, 'Predicted state': x_predicted_Logistic})
output.to_csv('Heart_Final_predicted_report.csv', index=False)