import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image

# --- CONFIG ---
IMG_SIZE = 256
LATENT_DIM = 100
BATCH_SIZE = 8
EPOCHS = 100
SAVE_INTERVAL = 5
GRAYSCALE = True  # Set to False if using RGB

DATA_PATHS = [
    "chest_xray/test/NORMAL",
    "chest_xray/test/PNEUMONIA",
]

# --- DATA LOADING ---
def load_images_from_folders(folders, img_size, max_images_per_folder=None):
    images = []
    for folder in folders:
        image_paths = glob(os.path.join(folder, "*"))
        for img_path in image_paths[:max_images_per_folder]:
            try:
                img = Image.open(img_path).convert("L" if GRAYSCALE else "RGB")
                img = img.resize((img_size, img_size))
                img = np.asarray(img) / 127.5 - 1.0
                images.append(img)
            except Exception as e:
                print(f"Skipped {img_path}: {e}")
    return np.array(images)
images = load_images_from_folders(DATA_PATHS, IMG_SIZE)
if GRAYSCALE:
    images = np.expand_dims(images, -1)

dataset = tf.data.Dataset.from_tensor_slices(images)
dataset = dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- GENERATOR ---
def build_generator():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(LATENT_DIM,)),
        layers.Dense(8 * 8 * 256),
        layers.Reshape((8, 8, 256)),

        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'),  # 16x16
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'),   # 32x32
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same'),   # 64x64
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(16, kernel_size=4, strides=2, padding='same'),   # 128x128
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(1 if GRAYSCALE else 3, kernel_size=4, strides=2, padding='same', activation='tanh'),  # 256x256
    ])
    return model

# --- DISCRIMINATOR ---
def build_discriminator():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1 if GRAYSCALE else 3)),
        layers.Conv2D(64, kernel_size=4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),

        layers.Conv2D(128, kernel_size=4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2D(256, kernel_size=4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# --- SETUP ---
generator = build_generator()
discriminator = build_discriminator()
cross_entropy = tf.keras.losses.BinaryCrossentropy()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# --- TRAINING STEP ---
@tf.function
def train_step(real_images):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])

    with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
        fake_images = generator(noise, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)

        disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + \
                    cross_entropy(tf.zeros_like(fake_output), fake_output)

        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    gradients_of_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gradients_of_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)

    discriminator_optimizer.apply_gradients(zip(gradients_of_disc, discriminator.trainable_variables))
    generator_optimizer.apply_gradients(zip(gradients_of_gen, generator.trainable_variables))

    return disc_loss, gen_loss

# --- GENERATE SAMPLE IMAGES ---
def generate_and_save_images(model, epoch, n=4):
    noise = tf.random.normal([n*n, LATENT_DIM])
    gen_imgs = model(noise, training=False)
    gen_imgs = (gen_imgs + 1) / 2.0  # Rescale to [0, 1]

    fig, axs = plt.subplots(n, n, figsize=(n, n))
    for i in range(n):
        for j in range(n):
            axs[i, j].imshow(gen_imgs[i*n + j, :, :, 0], cmap='gray' if GRAYSCALE else None)
            axs[i, j].axis('off')
    plt.tight_layout()
    os.makedirs("generated2", exist_ok=True)
    plt.savefig(f"generated2/sample_epoch_{epoch}.png")
    plt.close()

# --- TRAINING LOOP ---
for epoch in range(1, EPOCHS + 1):
    for real_batch in dataset:
        d_loss, g_loss = train_step(real_batch)

    print(f"Epoch {epoch}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")

    if epoch % SAVE_INTERVAL == 0:
        generate_and_save_images(generator, epoch)
        os.makedirs("generator2", exist_ok=True)
        os.makedirs("discriminator2", exist_ok=True)
        generator.save(f"generator2/generator_epoch_{epoch}.keras")
        discriminator.save(f"discriminator2/discriminator_epoch_{epoch}.keras")
