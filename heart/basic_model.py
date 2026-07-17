import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('cardio_train.csv', sep=';')

data['age'] = data['age'] // 365.25
data.drop(data[(data['height'] > data['height'].quantile(0.975)) | (data['height'] < data['height'].quantile(0.025))].index, inplace=True)
data.drop(data[(data['weight'] > data['weight'].quantile(0.975)) | (data['weight'] < data['weight'].quantile(0.025))].index, inplace=True)
data.drop(data[(data['ap_hi'] > data['ap_hi'].quantile(0.975)) | (data['ap_hi'] < data['ap_hi'].quantile(0.025))].index, inplace=True)
data.drop(data[(data['ap_lo'] > data['ap_lo'].quantile(0.975)) | (data['ap_lo'] < data['ap_lo'].quantile(0.025))].index, inplace=True)

data = data.rename(columns={'ap_hi': 'systolic_b_pressure'})
data = data.rename(columns={'ap_lo': 'diastolic_b_pressure'})
data = data.rename(columns={'gluc': 'glucose'})
data = data.rename(columns={'alco': 'alcohol'})
data = data.rename(columns={'active': 'physically_active'})
data = data.rename(columns={'cardio': 'cardio_disease'})


#Analysis

X = data.drop(columns=['id', 'cardio_disease'])
y = data['cardio_disease']


scaler_std = StandardScaler()
X_standardized = scaler_std.fit_transform(X)

feature_names = X.columns.tolist()
X_standardized_df = pd.DataFrame(X_standardized, columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(X_standardized_df, y, test_size=0.2, random_state=0)

model = GradientBoostingClassifier(random_state=0, max_depth=10, n_estimators=200)
model.fit(X_train,y_train)

print('Train Accuracy: ',model.score(X_train,y_train))
print('Test Accuracy: ', model.score(X_test,y_test))

joblib.dump(model, 'gradient_boosting_model.pkl')