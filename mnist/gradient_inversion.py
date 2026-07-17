import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

model = load_model('mnist.keras', compile=False)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_train = x_train.reshape(x_train.shape[0], 784)

def total_variation_loss(x):
    x_reshaped = tf.reshape(x, (28, 28))
    return tf.reduce_sum(tf.image.total_variation(tf.expand_dims(tf.expand_dims(x_reshaped, 0), -1)))

def get_initial_image(init_type, shape=(1, 784), x_data=None):
    if init_type == 'white':
        return tf.zeros(shape, dtype=tf.float32)
    elif init_type == 'grey':
        return tf.ones(shape, dtype=tf.float32) * 0.5
    elif init_type == 'black':
        return tf.ones(shape, dtype=tf.float32)
    elif init_type == 'random':
        return tf.random.uniform(shape, minval=0.0, maxval=1.0)
    elif init_type == 'average':
        avg_img = np.mean(x_data, axis=0).reshape(1, 784).astype(np.float32)
        return tf.convert_to_tensor(avg_img)
    else:
        raise ValueError("Unknown init_type")

def gradient_inversion(
    model, y_label,
    input_image,
    steps=500, lr=0.01,
    min_lr=5e-4, lr_decay=0.5,
    loss_threshold=0.3,
    patience=50, max_extra_steps=300
):
    y_batch = tf.convert_to_tensor(y_label, dtype=tf.int32)
    input_image = tf.Variable(input_image, dtype=tf.float32)

    best_loss = float('inf')
    best_image = None
    patience_counter = 0
    current_lr = lr

    for i in range(steps + max_extra_steps):
        with tf.GradientTape() as tape:
            predictions = model(input_image)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_batch, predictions))
            loss += 0.001 * tf.reduce_sum(tf.square(input_image))
            tv_weight = 0.01 if i < 200 else 0.0001
            loss += tv_weight * total_variation_loss(input_image)

        loss_val = loss.numpy()
        grads = tape.gradient(loss, input_image)
        input_image.assign_sub(current_lr * grads)
        input_image.assign(tf.clip_by_value(input_image, 0.0, 1.0))

        if loss_val < best_loss - 1e-5:
            best_loss = loss_val
            best_image = input_image.numpy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if current_lr > min_lr:
                current_lr *= lr_decay
                print(f"Reducing learning rate to {current_lr:.6f}")
                patience_counter = 0
            else:
                print(f"Early stopping at step {i}, best loss: {best_loss:.4f}")
                break

        if i % 50 == 0:
            print(f"Step {i}, Loss: {loss_val:.6f}, LR: {current_lr:.6f}")

        if loss_val < loss_threshold and i >= steps:
            print(f"Stopping at step {i} as loss is below threshold.")
            break

    return tf.convert_to_tensor(best_image)


init_types = ['white', 'grey', 'black', 'random', 'average']
num_classes = 10
all_results = {}

for init_type in init_types:
    print(f"\nInitialization: {init_type}")
    reconstructions = []
    for label in range(num_classes):
        print(f"Reconstructing class {label}")
        init_img = get_initial_image(init_type, shape=(1, 784), x_data=x_train)
        reconstructed = gradient_inversion(model, np.array([label]), init_img, steps=500, lr=0.01)
        reconstructions.append(reconstructed.numpy().reshape(28, 28))
    all_results[init_type] = reconstructions

fig, axs = plt.subplots(len(init_types), num_classes, figsize=(num_classes * 2, len(init_types) * 2))
for row, init_type in enumerate(init_types):
    for col in range(num_classes):
        axs[row, col].imshow(all_results[init_type][col], cmap='gray')
        axs[row, col].axis('off')
        if row == 0:
            axs[row, col].set_title(f"Class {col}")
    axs[row, 0].set_ylabel(init_type, rotation=90, size='large')

plt.suptitle("Reconstructed Images by Initialization and Class")
plt.tight_layout()
plt.show()
