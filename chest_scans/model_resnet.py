import os
import glob
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, callbacks, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

IMG_SIZE = 224
BATCH_SIZE = 32

main_path = "chest_xray/"
train_path = os.path.join(main_path, "train")

train_normal = glob.glob(train_path + "/NORMAL/*.jpeg")
train_pneumonia = glob.glob(train_path + "/PNEUMONIA/*.jpeg")

train_list = train_normal + train_pneumonia
labels = ['Normal'] * len(train_normal) + ['Pneumonia'] * len(train_pneumonia)
df_train = pd.DataFrame({"image": train_list, "class": labels})

train_df, val_df = train_test_split(df_train, test_size=0.20, random_state=seed, stratify=df_train['class'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                   zoom_range=0.1,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df, x_col='image', y_col='class',
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='binary',
    batch_size=BATCH_SIZE,
    seed=seed)

val_generator = val_datagen.flow_from_dataframe(
    val_df, x_col='image', y_col='class',
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='binary',
    batch_size=BATCH_SIZE,
    seed=seed)

test_path = os.path.join(main_path, "test")

test_normal = glob.glob(test_path + "/NORMAL/*.jpeg")
test_pneumonia = glob.glob(test_path + "/PNEUMONIA/*.jpeg")

test_list = test_normal + test_pneumonia
test_labels = ['Normal'] * len(test_normal) + ['Pneumonia'] * len(test_pneumonia)
df_test = pd.DataFrame({"image": test_list, "class": test_labels})

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
    df_test, x_col='image', y_col='class',
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='binary',
    batch_size=1,
    shuffle=False
)

base_model = tf.keras.applications.ResNet152V2(
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False
)
base_model.trainable = False  # Freeze base

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(inputs, outputs)

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[early_stopping, reduce_lr],
)

model.save('model_resnet.keras')

test_loss, test_acc = model.evaluate(test_generator, steps=len(df_test), verbose=1)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

label_map = {'Normal': 0, 'Pneumonia': 1}
Y_test = df_test['class'].map(label_map).values

test_generator.reset()
pred_probs = model.predict(test_generator, steps=len(test_generator), verbose=1)
pred_labels = (pred_probs > 0.5).astype(int).reshape(-1)

print("Calculated Accuracy:", accuracy_score(Y_test, pred_labels))

print(classification_report(Y_test, pred_labels, target_names=['Normal', 'Pneumonia']))

conf_matrix = confusion_matrix(Y_test, pred_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
