import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, callbacks, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
import seaborn as sns
import glob
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy_statement

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLIENTS = 5
EPOCHS = 3
ROUNDS = 3

main_path = "chest_xray/"
train_path = os.path.join(main_path, "train")

train_normal = glob.glob(train_path + "/NORMAL/*.jpeg")
train_pneumonia = glob.glob(train_path + "/PNEUMONIA/*.jpeg")

train_list = train_normal + train_pneumonia
labels = ['Normal'] * len(train_normal) + ['Pneumonia'] * len(train_pneumonia)
df_train = pd.DataFrame({"image": train_list, "class": labels})

train_df, val_df = train_test_split(df_train, test_size=0.20, random_state=seed, stratify=df_train['class'])

# Federated split
# client_data = np.array_split(train_df.sample(frac=1, random_state=seed), NUM_CLIENTS)

#non IID
def create_non_iid_clients(df, num_clients):
    pneumonia = df[df['class'] == 'Pneumonia']
    normal = df[df['class'] == 'Normal']
    
    pneumonia_clients = np.array_split(pneumonia, num_clients)
    normal_clients = np.array_split(normal, num_clients)

    client_dfs = []
    for i in range(num_clients):
        if i % 2 == 0:
            # More pneumonia
            client_df = pd.concat([pneumonia_clients[i], normal_clients[i].sample(frac=0.5, random_state=seed)])
        else:
            # More normal
            client_df = pd.concat([normal_clients[i], pneumonia_clients[i].sample(frac=0.5, random_state=seed)])
        client_dfs.append(client_df.sample(frac=1, random_state=seed))
    return client_dfs

client_data = create_non_iid_clients(train_df, NUM_CLIENTS)


# Data generators
datagen_args = dict(rescale=1./255, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1)
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_dataframe(
    val_df, x_col='image', y_col='class', target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='binary', batch_size=BATCH_SIZE, shuffle=False, seed=seed
)

# Base model definition
def create_model():
    base_model = tf.keras.applications.ResNet152V2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)

    optimizer = DPKerasAdamOptimizer(
        l2_norm_clip=1.0,
        noise_multiplier=1.1,
        num_microbatches=BATCH_SIZE, 
        learning_rate=5e-5
    )

    loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=False,  # sigmoid output
        reduction=tf.keras.losses.Reduction.NONE  # Needed for DP
    )

    model.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy'])
    return model


# Federated Averaging
def average_weights(weight_list):
    new_weights = list()
    for weights in zip(*weight_list):
        new_weights.append(np.mean(weights, axis=0))
    return new_weights

# Training Loop
global_model = create_model()

def dataframe_to_dataset(df, batch_size, augment=False):
    paths = df['image'].values
    labels = df['class'].map({'Normal': 0, 'Pneumonia': 1}).values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        def augment_fn(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.1)
            image = tf.image.random_contrast(image, 0.9, 1.1)
            return image, label
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.shuffle(1000, seed=seed)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


for rnd in range(ROUNDS):
    print(f"\nFederated Round {rnd+1}/{ROUNDS}")
    local_weights = []

    for cid, client_df in enumerate(client_data):
        print(f"  - Training on Client {cid + 1}")

        train_ds = dataframe_to_dataset(client_df, BATCH_SIZE, augment=True)

        client_model = create_model()
        client_model.set_weights(global_model.get_weights())

        client_model.fit(train_ds, epochs=EPOCHS, verbose=0)
        local_weights.append(client_model.get_weights())


    averaged_weights = average_weights(local_weights)
    global_model.set_weights(averaged_weights)

    val_loss, val_acc = global_model.evaluate(val_generator, verbose=0)
    print(f"✅ Round {rnd+1} Validation Accuracy: {val_acc:.4f}")

epsilon_statement = compute_dp_sgd_privacy_statement(
    number_of_examples=len(train_df),
    batch_size=BATCH_SIZE,
    num_epochs=ROUNDS * EPOCHS,
    noise_multiplier=1.1,
    delta=1e-5
)
print(epsilon_statement)


test_path = os.path.join(main_path, "test")
test_normal = glob.glob(test_path + "/NORMAL/*.jpeg")
test_pneumonia = glob.glob(test_path + "/PNEUMONIA/*.jpeg")
test_list = test_normal + test_pneumonia
test_labels = ['Normal'] * len(test_normal) + ['Pneumonia'] * len(test_pneumonia)
df_test = pd.DataFrame({"image": test_list, "class": test_labels})

test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    df_test, x_col='image', y_col='class', target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='binary', batch_size=1, shuffle=False
)

Y_test = df_test['class'].map({'Normal': 0, 'Pneumonia': 1}).values
pred_probs = global_model.predict(test_gen, verbose=1)
pred_labels = (pred_probs > 0.5).astype(int).reshape(-1)

print("Final Test Accuracy:", accuracy_score(Y_test, pred_labels))
print(classification_report(Y_test, pred_labels, target_names=['Normal', 'Pneumonia']))
global_model.save('dpsgd.keras')