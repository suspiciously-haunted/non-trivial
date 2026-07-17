import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model = load_model("model_fed.keras", compile=False)
IMG_SIZE = 224
target_label = 1  # 1 for Pneumonia, 0 for Normal

def total_variation_loss(x):
    return tf.reduce_sum(tf.image.total_variation(x))

def get_initial_image(init_type='random', shape=(1, IMG_SIZE, IMG_SIZE, 3), x_data=None):
    if init_type == 'white':
        return tf.zeros(shape, dtype=tf.float32)
    elif init_type == 'grey':
        return tf.ones(shape, dtype=tf.float32) * 0.5
    elif init_type == 'black':
        return tf.ones(shape, dtype=tf.float32)
    elif init_type == 'random':
        return tf.random.uniform(shape, minval=0.0, maxval=1.0)
    elif init_type == 'average':
        if x_data is None:
            raise ValueError("x_data must be provided for average initialization.")
        avg_img = np.mean(x_data, axis=0).astype(np.float32)
        return tf.convert_to_tensor(avg_img.reshape(1, IMG_SIZE, IMG_SIZE, 3))
    else:
        raise ValueError("Unknown init_type")

def gradient_inversion(
    model, target_label,
    input_image,
    steps=1000, lr=0.01,
    min_lr=5e-4, lr_decay=0.5,
    loss_threshold=0.2,
    patience=50, max_extra_steps=300
):
    input_image = tf.Variable(input_image, dtype=tf.float32)
    best_loss = float('inf')
    best_image = None
    patience_counter = 0
    current_lr = lr

    for i in range(steps + max_extra_steps):
        with tf.GradientTape() as tape:
            predictions = model(input_image)
            prob = tf.clip_by_value(predictions[0, 0], 1e-6, 1.0 - 1e-6)
            if target_label == 1:
                loss = -tf.math.log(prob)
            else:
                loss = -tf.math.log(1.0 - prob)

            loss += 0.001 * tf.reduce_sum(tf.square(input_image))
            loss += 0.01 * total_variation_loss(input_image)

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

        if i % 100 == 0:
            print(f"Step {i}, Loss: {loss_val:.6f}, LR: {current_lr:.6f}")

        if loss_val < loss_threshold and i >= steps:
            print(f"Stopping at step {i} as loss is below threshold.")
            break

    return tf.convert_to_tensor(best_image)

# Choose init type and reconstruct
init_type = 'random'
init_img = get_initial_image(init_type, shape=(1, IMG_SIZE, IMG_SIZE, 3))
reconstructed = gradient_inversion(model, target_label, init_img, steps=1000, lr=0.01)

# Show result
plt.imshow(reconstructed.numpy().squeeze())
plt.title(f"Reconstructed Image for Class {target_label} ('Pneumonia' if 1 else 'Normal')")
plt.axis('off')
plt.show()
