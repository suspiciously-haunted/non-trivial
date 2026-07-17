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
import seaborn as sns
import glob
import tenseal as ts

tf.get_logger().setLevel('ERROR')

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLIENTS = 3
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

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.generate_galois_keys()
context.generate_relin_keys()
context.global_scale = 2 ** 40

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
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    return model

def encrypt_weights(weights, context):
    encrypted = []
    for w in weights:
        flat = w.flatten()
        enc = ts.ckks_vector(context, flat.tolist())
        encrypted.append((enc, w.shape))
    return encrypted

def decrypt_and_average(encrypted_weights_list):
    num_clients = len(encrypted_weights_list)
    decrypted_avg = []

    for i in range(len(encrypted_weights_list[0])):
        enc_sum = encrypted_weights_list[0][i][0]
        for j in range(1, num_clients):
            enc_sum += encrypted_weights_list[j][i][0]

        enc_avg = enc_sum * (1.0 / num_clients) 
        decrypted = np.array(enc_avg.decrypt())
        shape = encrypted_weights_list[0][i][1]
        decrypted_avg.append(decrypted.reshape(shape))

    return decrypted_avg

global_model = create_model()

for rnd in range(ROUNDS):
    print(f"\nFederated Round {rnd+1}/{ROUNDS}")
    local_encrypted_weights = []

    for cid, client_df in enumerate(client_data):
        print(f"  - Training on Client {cid + 1}")

        train_gen = ImageDataGenerator(**datagen_args).flow_from_dataframe(
            client_df, x_col='image', y_col='class',
            target_size=(IMG_SIZE, IMG_SIZE), class_mode='binary',
            batch_size=BATCH_SIZE, shuffle=True, seed=seed
        )

        client_model = create_model()
        client_model.set_weights(global_model.get_weights())

        client_model.fit(train_gen, epochs=EPOCHS, verbose=0)

        encrypted_weights = encrypt_weights(client_model.get_weights(), context)
        local_encrypted_weights.append(encrypted_weights)

    averaged_weights = decrypt_and_average(local_encrypted_weights)
    global_model.set_weights(averaged_weights)

    val_loss, val_acc = global_model.evaluate(val_generator, verbose=0)
    print(f"✅ Round {rnd+1} Validation Accuracy: {val_acc:.4f}")


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
global_model.save('tenseal.keras')
# Confusion Matrix
cm = confusion_matrix(Y_test, pred_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
