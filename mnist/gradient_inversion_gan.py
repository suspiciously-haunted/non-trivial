import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer

os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(42)
np.random.seed(42)

latent_dim = 100
steps = 500
max_steps = 2000
loss_threshold = 1.5

G = load_model("gan_generator.keras", compile=False)
model = tf.keras.models.load_model("non-tenseal.keras", compile=False)


(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0


generated_images = []
original_images = []

for target_label in range(10):
    print(f"\n=== Generating for label {target_label} ===")
    
    
    original_img = x_train[y_train == target_label][0]
    original_images.append(original_img)


    z = tf.Variable(tf.random.normal([1, latent_dim], stddev=0.5), trainable=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    step = 0
    while step < max_steps:
        with tf.GradientTape() as tape:
            generated_img = G(z)
            processed_img = (generated_img + 1) / 2 
            flat_img = tf.reshape(processed_img, [1, 784])
            logits = model(flat_img)

            cls_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[target_label], logits=logits)
            )
            reg_loss = 0.00005 * tf.reduce_sum(tf.square(z))
            loss = cls_loss + reg_loss

        grads = tape.gradient(loss, [z])
        optimizer.apply_gradients(zip(grads, [z]))

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.numpy():.4f}")
            if loss.numpy() < loss_threshold and step >= steps:
                print(f"Stopping early at step {step}, loss = {loss.numpy():.4f}")
                break
        step += 1

    final_img = ((G(z).numpy().squeeze() + 1) / 2).reshape(28, 28)  
    generated_images.append(final_img)


fig, axes = plt.subplots(2, 10, figsize=(15, 3))

for i in range(10):
    axes[0, i].imshow(generated_images[i], cmap='gray')
    axes[0, i].set_title(f'Label {i}')
    axes[0, i].axis('off')

    axes[1, i].imshow(original_images[i], cmap='gray')
    axes[1, i].axis('off')

plt.suptitle("Top: Reconstructed | Bottom: Original MNIST", fontsize=14)
plt.tight_layout()
plt.show()
