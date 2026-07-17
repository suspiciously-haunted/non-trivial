import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
from tensorflow.keras.losses import BinaryCrossentropy
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
joblib.dump(scaler, "scaler_dpsgd.pkl")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Config ---
NUM_CLIENTS = 50
NUM_ROUNDS = 5
LOCAL_EPOCHS = 5

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

    ds = tf.data.Dataset.from_tensor_slices((X_client, y_client))
    ds = ds.shuffle(1024).batch(32, drop_remainder=True)
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

dp_optimizer = DPKerasAdamOptimizer(
            l2_norm_clip=1.0,           # Clip gradient norm to 1.0
            noise_multiplier=1.1,       # Add Gaussian noise scaled to this
            num_microbatches=32,        # Must match batch size
            learning_rate=0.001
        )

loss_fn = BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

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

            # Save gradients and corresponding inputs/labels (optional for eval)
            if round_num == NUM_ROUNDS-1 and client_id == 0:
                print(f"Saving leaked gradients from Client 0, Round {NUM_ROUNDS-1}...")
                np.savez("leaked_data_dpsgd.npz",
                        **{f"grad_{i}": g.numpy() for i, g in enumerate(grads)},
                        inputs=x_batch.numpy(),
                        labels=y_batch.numpy())

        local_model.compile(optimizer=dp_optimizer, loss=loss_fn, metrics=['accuracy'])

        # local_model.compile(optimizer=dp_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        local_model.fit(client_ds, epochs=LOCAL_EPOCHS, verbose=0)

        # Step 3: Store trained weights
        client_weights.append(local_model.get_weights())

    # Step 4: Average weights (FedAvg)
    new_weights = []
    for weights_tuple in zip(*client_weights):
        new_weights.append(np.mean(weights_tuple, axis=0))

    # Step 5: Update global model
    global_model.set_weights(new_weights)

    # Evaluate
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)
    loss, acc = global_model.evaluate(test_ds, verbose=0)
    print(f"Global Accuracy = {acc:.4f}")

global_model.save("dpsgd.keras")
