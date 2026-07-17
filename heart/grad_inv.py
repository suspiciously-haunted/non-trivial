import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
import os
import random

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# --- Load model and leaked gradients ---
model = load_model("fed_test.keras", compile=False)
input_dim = model.input_shape[1]

leak = np.load("leaked_data_test.npz")
true_input = leak["inputs"][0:]
true_label = leak["labels"][0:] 

scaler = joblib.load("scaler_test.pkl")

target_grads = [leak[f"grad_{i}"] for i in range(len([k for k in leak.keys() if k.startswith("grad_")]))]

# --- Initialize dummy input/label ---
dummy_x = tf.Variable(tf.random.normal((1, input_dim)), trainable=True)
dummy_y = tf.Variable(tf.random.uniform((1, 1), minval=0, maxval=1), trainable=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
loss_history = []

# --- DLG Optimization ---
for step in range(2000):
    with tf.GradientTape() as outer_tape:
        with tf.GradientTape() as inner_tape:
            pred = model(dummy_x, training=True)
            loss = tf.keras.losses.binary_crossentropy(tf.sigmoid(dummy_y), pred)
        grads = inner_tape.gradient(loss, model.trainable_variables)

        grad_loss = tf.add_n([
            tf.reduce_mean((g - tf.convert_to_tensor(tg)) ** 2)
            for g, tg in zip(grads, target_grads)
        ])

    dummy_grads = outer_tape.gradient(grad_loss, [dummy_x, dummy_y])
    if None in dummy_grads:
        raise ValueError("Missing gradient. Check variable initialization.")

    optimizer.apply_gradients(zip(dummy_grads, [dummy_x, dummy_y]))
    loss_history.append(grad_loss.numpy())

    if step % 100 == 0:
        print(f"Step {step}: Grad MSE = {grad_loss.numpy():.6f}")

# --- Output results ---
print("Reconstructed input (shape):", dummy_x.shape)

reconstructed_input = dummy_x.numpy()
reconstructed_orig = scaler.inverse_transform(reconstructed_input)
real_orig = scaler.inverse_transform(true_input)

reconstructed_orig[0, 1] = np.clip(round(reconstructed_orig[0, 1]), 1, 2)

for col in [6, 7]:
    reconstructed_orig[0, col] = np.clip(round(reconstructed_orig[0, col]), 1, 3)


for col in [8, 9, 10]:
    reconstructed_orig[0, col] = np.clip(round(reconstructed_orig[0, col]), 0, 1)

print("Reconstructed input:")
np.set_printoptions(suppress=True, precision=0)
print(reconstructed_orig)
pred_label = int(tf.sigmoid(dummy_y).numpy().flatten()[0] > 0.5)
print("Predicted binary label:", pred_label)

# print("\nReal input:")
# print(real_orig)
# print("\nOriginal label:", true_label)

avg_input = np.mean(leak["inputs"], axis=0, keepdims=True)  # shape: (1, num_features)
avg_label = np.mean(leak["labels"], keepdims=True)          # shape: (1,)
avg_input_original_scale = scaler.inverse_transform(avg_input)

print(f"Average Input:")
print(avg_input_original_scale)
print("Average Label:", int(avg_label[0] > 0.5))