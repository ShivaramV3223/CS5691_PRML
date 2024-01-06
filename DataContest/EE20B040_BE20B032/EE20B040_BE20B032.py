#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns


# In[74]:


# Load all the files given to us

csv_files = ['bookings.csv', 'bookings_data.csv', 'customer_data.csv', 'hotels_data.csv', 'payments_data.csv']
train_file = 'train_data.csv'
submission_file = 'sample_submission_5.csv'

csv_data = [pd.read_csv('./data/'+csv_file) for csv_file in csv_files]
dataframes = [pd.DataFrame(csv) for csv in csv_data]
train_data = pd.read_csv('./data/'+train_file)
train_df = pd.DataFrame(train_data)
submission_data = pd.read_csv('./data/'+submission_file)
submission_df = pd.DataFrame(submission_data)


# In[75]:


# Preprocess the given dataframes

# 1. Drop missing values in hotels_data
dataframes[3].dropna(inplace=True)

# 2. Fill the missing values of booking_approved_at
dataframes[0]['booking_create_timestamp'] = pd.to_datetime(dataframes[0]['booking_create_timestamp'])
dataframes[0]['booking_approved_at'] = pd.to_datetime(dataframes[0]['booking_approved_at'])
dataframes[0]['booking_checkin_customer_date'] = pd.to_datetime(dataframes[0]['booking_checkin_customer_date'])
req_series = dataframes[0].loc[dataframes[0]['booking_approved_at'].isnull()]['booking_create_timestamp'] - timedelta(seconds=1)
dataframes[0]['booking_approved_at'].fillna(value=req_series, inplace=True)
req_series = dataframes[0].loc[dataframes[0]['booking_checkin_customer_date'].isnull()]['booking_create_timestamp']
dataframes[0]['booking_checkin_customer_date'].fillna(value=req_series, inplace=True)

# 3. Merge all dataframes into one master dataframe
master_df = pd.merge(dataframes[0], dataframes[1], how='outer', on='booking_id')
master_df = pd.merge(master_df, dataframes[2], how='outer', on='customer_id')
master_df = pd.merge(master_df, dataframes[3], how='outer', on='hotel_id')
master_df = pd.merge(master_df, dataframes[4], how='outer', on='booking_id')

# 4. Get booking subrequests and Drop all duplicate booking ids
master_df['booking_sequence_id'].fillna(1, inplace=True)
master_df.drop_duplicates(subset='booking_id', keep='last', inplace=True)

# 5. Fill hotel and payment attributes with the mean values 
mode_hc = master_df['hotel_category'].mode()[0]
master_df['hotel_category'].fillna(mode_hc, inplace=True)
master_df['hotel_name_length'].fillna(int(master_df['hotel_name_length'].mean()), inplace=True)
master_df['hotel_description_length'].fillna(int(master_df['hotel_description_length'].mean()), inplace=True)
master_df['hotel_photos_qty'].fillna(int(master_df['hotel_photos_qty'].mean()), inplace=True)
master_df['price'].fillna(master_df['price'].mean(), inplace=True)
master_df['agent_fees'].fillna(master_df['agent_fees'].mean(), inplace=True)

# 6. Fill in the one payment value with price + agent fees
master_df.at[53487, 'payment_type'] = 'not_defined'
master_df.at[53487, 'payment_installments'] = 1

# 7. Create the columns where difference between the datetime attributes are given
master_df['timediff_ac'] = (master_df['booking_approved_at'] - master_df['booking_create_timestamp']).dt.seconds
master_df['timediff_cc'] = (master_df['booking_checkin_customer_date'] - master_df['booking_create_timestamp']).dt.seconds

# 8. Drop all the ids and unnceseary columns
master_df.drop(['customer_unique_id', 'seller_agent_id', 
                'customer_id', 'booking_expiry_date', 'hotel_id', 
               'payment_sequential', 'booking_approved_at', 'booking_create_timestamp', 
               'booking_checkin_customer_date', 'payment_value'], axis=1, inplace=True)

# 9. MinMax Scaling for all the attributes that require scaling
cols = ['price', 'agent_fees', 'timediff_ac', 'timediff_cc']
scaler = RobustScaler()
master_df[cols] = scaler.fit_transform(master_df[cols])

# 10. Merge with training dataframes to get the training data
training_df = pd.merge(master_df, train_df, on='booking_id', how='inner')
training_df.drop(['booking_id'], axis=1, inplace=True)
testing_df = pd.merge(master_df, submission_df, on='booking_id', how='inner')
testing_df.drop(['booking_id'], axis=1, inplace=True)

# 11. Create encoders for the string attributes
countries = list(set(master_df['country'].dropna()))
statusses = list(set(master_df['booking_status'].dropna()))
payment_types = list(set(master_df['payment_type'].dropna()))
encoder_country = LabelEncoder()
encoder_status = LabelEncoder()
encoder_ptype = LabelEncoder()
encoder_country.fit(countries)
encoder_status.fit(statusses)
encoder_ptype.fit(payment_types)

# 12. Encode the columns respectively
training_df['country'] = encoder_country.transform(training_df['country'])
training_df['booking_status'] = encoder_status.transform(training_df['booking_status'])
training_df['payment_type'] = encoder_ptype.transform(training_df['payment_type'])
testing_df['country'] = encoder_country.transform(testing_df['country'])
testing_df['booking_status'] = encoder_status.transform(testing_df['booking_status'])
testing_df['payment_type'] = encoder_ptype.transform(testing_df['payment_type'])

# 13. Create the training and testing datasets
Y_train = training_df['rating_score'].to_numpy()
X_train = training_df.drop(['rating_score'], axis=1).to_numpy()
X_test = testing_df.drop(['rating_score'], axis=1).to_numpy()

# 15. Apply PCA
pca = PCA(n_components=4)
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)

# Sanity check
# print(training_df.shape)
# print(testing_df.shape)


# In[71]:


# Create a model class using which we predict the ratings

from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

class Model:
    def __init__(self):
        self.clf1 = LinearRegression()
        self.clf2 = AdaBoostRegressor(n_estimators=500, learning_rate=0.1)
        self.clf3 = RandomForestRegressor(n_estimators=500, max_depth=7)
        self.classifier = VotingRegressor([('lin', self.clf1), ('ada', self.clf2), ('rf', self.clf3)])
    def train(self, X_train, Y_train):
        self.classifier.fit(X_train, Y_train)
    def predict(self, X_test):
        return self.classifier.predict(X_test)


# In[69]:


# Train the model on training data after preprocessing

def train_model(X, Y, val_size=0.2, shuffle=True, random_state=42):
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_size,
                                                      shuffle=shuffle, random_state=random_state)
    # Train the model
    model = Model()
    model.train(X_train, Y_train)
    # Return the validation error
    Y_pred = model.predict(X_val)
    val_err = mse(Y_val, Y_pred)
    return model, val_err

# Predict the submission data after preprocessing 

def predict_rating(model, X_test, test_df, write=False):
    Y_test = model.predict(X_test)
    test_df['rating_score'] = Y_test
    if write:
        test_df.to_csv('submission.csv', index=False)
    return Y_test


# In[76]:


# Implement the functions 
import time
start = time.time()
model, val_err = train_model(X_train, Y_train, val_size=0.2, random_state=77)
end = time.time()
print(f'validation error: {val_err:.3f}')
print(f'time elapsed to train: {end-start:.3f}')
Y_pred = predict_rating(model, X_test, submission_df, write=True)

