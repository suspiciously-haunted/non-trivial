import os
import random
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

IMG_SIZE = 224
BATCH_SIZE = 32

model = load_model('model_fed.keras', compile=False)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    return img_array

main_path = "chest_xray/"

# Training data (used for members)
train_path = os.path.join(main_path, "train")
train_normal = glob.glob(train_path + "/NORMAL/*.jpeg")
train_pneumonia = glob.glob(train_path + "/PNEUMONIA/*.jpeg")
train_list = train_normal + train_pneumonia
train_labels = ['Normal'] * len(train_normal) + ['Pneumonia'] * len(train_pneumonia)
df_train = pd.DataFrame({"image": train_list, "class": train_labels})
train_df, val_df = train_test_split(df_train, test_size=0.20, random_state=seed, stratify=df_train['class'])

# Test data (used for non-members)
test_path = os.path.join(main_path, "test")
test_normal = glob.glob(test_path + "/NORMAL/*.jpeg")
test_pneumonia = glob.glob(test_path + "/PNEUMONIA/*.jpeg")
test_list = test_normal + test_pneumonia
test_labels = ['Normal'] * len(test_normal) + ['Pneumonia'] * len(test_pneumonia)
df_test = pd.DataFrame({"image": test_list, "class": test_labels})

label_map = {'Normal': 0, 'Pneumonia': 1}

def load_images_from_df(df, label_map):
    x = np.array([preprocess_image(path) for path in df['image']])
    y = df['class'].map(label_map).values
    return x, y

member_df = train_df.sample(n=200, random_state=seed)
non_member_df = df_test.sample(n=200, random_state=seed)

x_member, y_member = load_images_from_df(member_df, label_map)
x_non_member, y_non_member = load_images_from_df(non_member_df, label_map)

y_member = y_member.reshape(-1, 1).astype('float32')
y_non_member = y_non_member.reshape(-1, 1).astype('float32')

member_probs = model.predict(x_member, verbose=0)
non_member_probs = model.predict(x_non_member, verbose=0)

loss_fn = BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

member_losses = loss_fn(y_member, member_probs).numpy()
non_member_losses = loss_fn(y_non_member, non_member_probs).numpy()

member_scores = -member_losses
non_member_scores = -non_member_losses

all_scores = np.concatenate([member_scores, non_member_scores])
membership_labels = np.array([1] * len(member_scores) + [0] * len(non_member_scores))

mia_auc = roc_auc_score(membership_labels, all_scores)
print(f"🔍 MIA ROC-AUC using Loss: {mia_auc:.4f}")

plt.hist(member_probs, bins=30, alpha=0.5, label='Member')
plt.hist(non_member_probs, bins=30, alpha=0.5, label='Non-member')
plt.axvline(x=0.5, color='red', linestyle='--', label='0.5 Threshold')
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.title("Membership Inference Confidence Distributions")
plt.legend()
plt.show()
