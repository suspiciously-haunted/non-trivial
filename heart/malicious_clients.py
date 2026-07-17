import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import random
import os

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# --- Load and preprocess your data ---
data = pd.read_csv('cardio_train.csv', sep=';')
data['age'] = data['age'] // 365.25
data = data.rename(columns={
    'ap_hi': 'systolic_b_pressure',
    'ap_lo': 'diastolic_b_pressure',
    'gluc': 'glucose',
    'alco': 'alcohol',
    'active': 'physically_active',
    'cardio': 'cardio_disease'
})

for col in ['height', 'weight', 'systolic_b_pressure', 'diastolic_b_pressure']:
    data = data[(data[col] > data[col].quantile(0.025)) & (data[col] < data[col].quantile(0.975))]
data.info()
X = data.drop(columns=['id', 'cardio_disease'])
y = data['cardio_disease']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler_test.pkl")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Config ---
NUM_CLIENTS = 100
NUM_ROUNDS = 5
LOCAL_EPOCHS = 5
MALICIOUS_CLIENTS = [i for i in range(NUM_CLIENTS) if (i % 5 != 0 and i % 5 != 2)]


# --- Non-IID client partitioning ---
client_data = []

class_0_indices = np.where(y_train == 0)[0]
class_1_indices = np.where(y_train == 1)[0]

np.random.shuffle(class_0_indices)
np.random.shuffle(class_1_indices)

split_class_0 = np.array_split(class_0_indices, NUM_CLIENTS)
split_class_1 = np.array_split(class_1_indices, NUM_CLIENTS)

for i in range(NUM_CLIENTS):
    if i < NUM_CLIENTS // 2:
        indices = np.concatenate([split_class_0[i], split_class_1[i][:len(split_class_1[i]) // 4]])
    else:
        indices = np.concatenate([split_class_1[i], split_class_0[i][:len(split_class_0[i]) // 4]])

    np.random.shuffle(indices)
    X_client = X_train[indices]
    y_client = y_train.values[indices]

    # Apply malicious behavior
    if i in MALICIOUS_CLIENTS:
        y_client = 1 - y_client  # Flip 0 to 1 and 1 to 0

    ds = tf.data.Dataset.from_tensor_slices((X_client, y_client))
    ds = ds.shuffle(1024).batch(1)
    client_data.append(ds)

# --- Define global model architecture ---
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_scaled.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

global_model = build_model()
global_weights = global_model.get_weights()
optimizer = tf.keras.optimizers.Adam()

global_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

for round_num in range(NUM_ROUNDS):
    print(f"\n--- Round {round_num + 1} ---")
    client_weights = []

    for client_id, client_ds in enumerate(client_data):
        # Step 1: Clone the global model
        local_model = build_model()
        local_model.set_weights(global_model.get_weights())
        # Step 2: Manually compute gradients on the first batch of the client
        for x_batch, y_batch in client_ds.take(1):
            with tf.GradientTape() as tape:
                y_pred = local_model(x_batch, training=True)
                loss = tf.keras.losses.binary_crossentropy(tf.reshape(y_batch, (-1, 1)), y_pred)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, local_model.trainable_variables)

        # Step 2: Compile and train on local data
        local_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        local_model.fit(client_ds, epochs=LOCAL_EPOCHS, verbose=0)

        # Step 3: Store trained weights
        client_weights.append(local_model.get_weights())

    # Step 4: Average weights (FedAvg)
    # new_weights = []
    # for weights_tuple in zip(*client_weights):
    #     new_weights.append(np.mean(weights_tuple, axis=0))

    def trimmed_mean(updates, trim_ratio=0.1):
        new_weights = []
        for layer_updates in zip(*updates):  # for each layer
            stacked = np.stack(layer_updates, axis=0)  # shape: (num_clients, ...)
            k = int(trim_ratio * stacked.shape[0])
            sorted_weights = np.sort(stacked, axis=0)
            trimmed_weights = sorted_weights[k:-k]  # remove top and bottom k clients
            layer_mean = np.median(trimmed_weights, axis=0)
            new_weights.append(layer_mean)
        return new_weights
    # robust aggregation
    new_weights = trimmed_mean(client_weights, trim_ratio=0.1)


    # Step 5: Update global model
    global_model.set_weights(new_weights)

    # Evaluate
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64)
    loss, acc = global_model.evaluate(test_ds, verbose=0)
    print(f"Global Accuracy = {acc:.4f}")
