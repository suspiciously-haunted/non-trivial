import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
from tensorflow.keras.utils import load_img, img_to_array

os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(42)
np.random.seed(42)

latent_dim = 100
steps = 500
max_steps = 2000
loss_threshold = 0.5

G = load_model("GAN_generator_2.keras", compile=False)
model = tf.keras.models.load_model(
    "tenseal.keras",
    custom_objects={'DPKerasAdamOptimizer': DPKerasAdamOptimizer},
    compile=False
)

generated_images = []
original_images = []

main_path = "chest_xray/"
train_path = os.path.join(main_path, "train")
train_normal = glob.glob(train_path + "/NORMAL/*.jpeg")
train_pneumonia = glob.glob(train_path + "/PNEUMONIA/*.jpeg")

for img_path in [train_normal[0], train_pneumonia[0]]:
    img = load_img(img_path, color_mode='grayscale', target_size=(224, 224))
    img_array = img_to_array(img).squeeze() / 255.0 
    original_images.append(img_array)

for target_label in [0, 1]:
    print(f"\n=== Generating for label {target_label} ===")

    z = tf.Variable(tf.random.normal([1, latent_dim], stddev=0.5), trainable=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    step = 0
    while step < max_steps:
        with tf.GradientTape() as tape:
            generated_img = G(z)
            processed_img = (generated_img + 1) / 2 
            processed_img = tf.image.resize(processed_img, [224, 224]) 
            processed_img = tf.image.grayscale_to_rgb(processed_img)
            flat_img = tf.reshape(processed_img, [1, 224, 224, 3]) 
            logits = model(flat_img)
            label_tensor = tf.constant([[target_label]], dtype=tf.float32)
            cls_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(label_tensor, logits)
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

    final_img = (G(z).numpy().squeeze() + 1) / 2
    generated_images.append(final_img)


fig, axes = plt.subplots(2, 2, figsize=(6, 3))

for i in [0, 1]:
    axes[0, i].imshow(generated_images[i], cmap='gray')
    axes[0, i].set_title(f'Label {i}')
    axes[0, i].axis('off')

    axes[1, i].imshow(original_images[i], cmap='gray')
    axes[1, i].axis('off')

plt.suptitle("Top: Reconstructed | Bottom: Original", fontsize=14)
plt.tight_layout()
plt.show()
