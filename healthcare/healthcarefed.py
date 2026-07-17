import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LeakyReLU
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.regularizers import l2

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

NUM_CLIENTS = 5
NUM_ROUNDS = 3
EPOCHS = 10

skf = StratifiedKFold(n_splits=NUM_CLIENTS, shuffle=True, random_state=42)

client_X_splits = []
client_y_splits = []

for _, client_index in skf.split(X_train, y_train):
    client_X_splits.append(X_train.iloc[client_index])
    client_y_splits.append(y_train.iloc[client_index])

for i, y_client in enumerate(client_y_splits):
    print(f"Client {i+1} class distribution after stratified split:")
    print(y_client.value_counts(normalize=True))
    print("-" * 40)

input_dim = train_df.shape[1] - 1
num_classes = len(df["Medical Condition"].unique())

def build_model():
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.1),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

global_model = build_model()

# --- Federated training ---
for round_num in range(NUM_ROUNDS):
    print(f"\n--- Federated Round {round_num + 1} ---")
    client_weights = []
    for i in range(NUM_CLIENTS):
        model = build_model()
        model.set_weights(global_model.get_weights())
        
        model.fit(client_X_splits[i], client_y_splits[i], epochs=EPOCHS, batch_size=16, verbose=1)
        client_weights.append(model.get_weights())
        y_pred = np.argmax(model.predict(X_test), axis=1)
        acc = accuracy_score(y_test, y_pred)
        print(f"Client {i+1} model accuracy on global test set: {acc:.4f}")

    # for i in range(NUM_CLIENTS):
    #     model = build_model()
    #     model.set_weights(global_model.get_weights())
        
    #     X_client = client_X_splits[i]
    #     y_client = client_y_splits[i]

    #     model.fit(X_client, y_client, epochs=EPOCHS, batch_size=16, verbose=1)
    #     client_weights.append(model.get_weights())
    #     print(f"Client {i+1} finished training.")

    new_weights = []
    for weights in zip(*client_weights):
        new_weights.append(np.mean(weights, axis=0))

    global_model.set_weights(new_weights)
    print("Global model updated.")

# --- Evaluation ---
y_pred_probs = global_model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test, y_pred)
print(f"\nFederated Test Accuracy: {acc:.4f}")

decoded_preds = label_encoder.inverse_transform(y_pred)
decoded_true = label_encoder.inverse_transform(y_test)

results = pd.DataFrame({
    "Predicted Condition": decoded_preds,
    "Actual Condition": decoded_true
})

print(results.head(10))

global_model.save('healthcare_model2.keras')
print("Model saved as healthcare_model2.keras")