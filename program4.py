# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf

#----------------Data Preprocessing--------------------

# Importing the dataset
dataset = pd.read_csv('program4input.txt')

dataset.head()
dataset.info()

# make TotalCharges column a numeric value
dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors = 'coerce')

# calculate values to fill nulls in TotalCharges
dataset['TotalCharges'].fillna(value=dataset['tenure'] * dataset['MonthlyCharges'], inplace=True)

# make the SeniorCitizen column into an object
dataset['SeniorCitizen'] = dataset['SeniorCitizen'].apply(lambda x: 'Yes' if x == 1 else 'No')


# make Churn column a numeric value
def churn_to_numeric(value):
    if value.lower() == 'yes':
        return 1
    return 0

dataset['Churn'] = dataset['Churn'].apply(churn_to_numeric)

X = dataset.drop(['customerID', 'Churn'], axis=1)
y = dataset['Churn']

# split into training, text, and validation
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.17647, random_state=1)


tenure = tf.feature_column.numeric_column('tenure')
monthly_charges = tf.feature_column.numeric_column('MonthlyCharges')
total_charges = tf.feature_column.numeric_column('TotalCharges')

col_unique_val_counts = []
cat_columns = []
for col in X.columns:
    if X[col].dtype.name != 'object':
        continue
    unique_vals = X[col].unique()
    col_unique_val_counts.append(len(unique_vals))
    cat_columns.append(col)
    print(col, "->",unique_vals)
    
cat_cols = [tf.feature_column.categorical_column_with_hash_bucket(col, hash_bucket_size=size) 
            for col, size in zip(cat_columns, col_unique_val_counts)]

num_cols = [tenure, monthly_charges, total_charges]
feature_columns = num_cols + cat_cols

# Build Linear Classification Model
n_classes = 2 # churn Yes or No
batch_size = 100

sess = tf.Session()

# initialize all global variables  
sess.run(tf.global_variables_initializer())

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=batch_size,num_epochs=1000, shuffle=True)

linear_model= tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=n_classes)

linear_model.train(input_fn=input_func, steps=10000) 

eval_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      y=y_test,
      batch_size=batch_size,
      num_epochs=1,
      shuffle=False)

eval_result = linear_model.evaluate(eval_input_func)

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

# run test set and display results

pred_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      batch_size=batch_size,
      num_epochs=1,
      shuffle=False)

preds = linear_model.predict(pred_input_func)

predictions = [p['class_ids'][0] for p in preds]

from sklearn.metrics import classification_report

target_names = ['No', 'Yes']

print(classification_report(y_test, predictions, target_names=target_names))

# run validation set and display results

pred_input_func_val = tf.estimator.inputs.pandas_input_fn(
      x=X_val,
      batch_size=batch_size,
      num_epochs=1,
      shuffle=False)

preds = linear_model.predict(pred_input_func_val)

predictions = [p['class_ids'][0] for p in preds]

target_names = ['No', 'Yes']

print(classification_report(y_test, predictions, target_names=target_names))

# Prints the name of the variable alongside its value.
tvars = linear_model.get_variable_names() 
tvars_vals = [linear_model.get_variable_value(name) for name in tvars]

for var, val in zip(tvars, tvars_vals):
    print(var, val)  
    
sess.close()





