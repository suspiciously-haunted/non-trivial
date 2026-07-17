import numpy as np
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import Input
from sklearn.model_selection import StratifiedKFold


df = pd.read_csv('healthcare_dataset.csv')
df.drop(columns=['Name', 'Date of Admission', 'Insurance Provider', 'Room Number', 'Blood Type', 'Discharge Date'], inplace=True)
df = df.ffill()

df['Billing Amount'] = abs(df['Billing Amount'])
df['Billing Amount'] = df['Billing Amount'].apply(lambda x: np.log1p(x))

df = pd.get_dummies(df, columns=['Gender', 'Doctor', 'Hospital', 'Admission Type', 'Medication', 'Test Results'])

label_encoder = LabelEncoder()
df["Medical Condition"] = label_encoder.fit_transform(df["Medical Condition"])

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

X_train = train_df.drop("Medical Condition", axis=1).copy()
y_train = train_df["Medical Condition"]

X_test = test_df.drop("Medical Condition", axis=1).copy()
y_test = test_df["Medical Condition"]

scaler = StandardScaler()
X_train['Age'] = scaler.fit_transform(X_train[['Age']])
X_test['Age'] = scaler.transform(X_test[['Age']])
X_train['Billing Amount'] = scaler.fit_transform(X_train[['Billing Amount']])
X_test['Billing Amount'] = scaler.transform(X_test[['Billing Amount']])

model = load_model('healthcare_model.keras')

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

def inference_attack(model, client_data, target_class):
    predictions = model.predict(client_data)
    predicted_class = np.argmax(predictions, axis=1)
    
    count = np.sum(predicted_class == target_class)
    print(f"{count}/{len(client_data)} inputs predicted as class {target_class}")
    
    if count > 0:
        print("Possible inference success!")
    else:
        print("No inference success.")

target_label = label_encoder.transform(["Diabetes"])[0]
target_data = X_test[y_test == target_label][:10]
inference_attack(model, target_data, target_class=label_encoder.transform(["Diabetes"])[0])