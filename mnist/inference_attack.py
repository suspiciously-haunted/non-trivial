import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_auc_score

model = load_model('tenseal.keras', compile=False)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = x_train.reshape(-1, 784)
x_test = x_test.astype("float32") / 255.0
x_test = x_test.reshape(-1, 784)

# Simulate training data
# In real MIA, you'd track which data was used for training. Here, we'll simulate.
x_train_simulated = x_train[:2000]
y_train_simulated = y_train[:2000]

x_non_member = x_test[2000:4000]
y_non_member = y_test[2000:4000]

# Convert labels to one-hot
y_member_oh = to_categorical(y_train_simulated, num_classes=10)
y_non_member_oh = to_categorical(y_non_member, num_classes=10)

member_probs = model.predict(x_train_simulated, verbose=0)
non_member_probs = model.predict(x_non_member, verbose=0)

loss_fn = CategoricalCrossentropy(reduction='none')
member_losses = loss_fn(y_member_oh, member_probs).numpy()
non_member_losses = loss_fn(y_non_member_oh, non_member_probs).numpy()

member_scores = -member_losses
non_member_scores = -non_member_losses

all_scores = np.concatenate([member_scores, non_member_scores])
membership_labels = np.array([1]*len(member_scores) + [0]*len(non_member_scores))

mia_auc = roc_auc_score(membership_labels, all_scores)
print(f"🔍 MIA ROC-AUC using Loss (MNIST): {mia_auc:.4f}")

plt.hist(member_scores, bins=30, alpha=0.5, label='Members (−Loss)')
plt.hist(non_member_scores, bins=30, alpha=0.5, label='Non-Members (−Loss)')
plt.axvline(x=0, color='red', linestyle='--', label='0 Loss Line')
plt.xlabel("−Categorical Crossentropy Loss")
plt.ylabel("Count")
plt.title("MIA on MNIST using −Loss")
plt.legend()
plt.tight_layout()
plt.show()
