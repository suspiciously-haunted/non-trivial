import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from federated_learning import federated_learning
import time
import psutil
import os
import warnings
warnings.filterwarnings("ignore")

start = time.process_time()
process = psutil.Process(os.getpid())
cpu_before = psutil.cpu_percent(interval=1)
mem_before = process.memory_info().rss / (1024 * 1024)

final_model = federated_learning(num_rounds=10, num_clients=0, num_malicious=5, num_epochs=10)

(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
x_test = x_test.reshape(x_test.shape[0], 784) 

test_loss, test_accuracy = final_model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

y_pred = final_model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

end = time.process_time()
cpu_after = psutil.cpu_percent(interval=1)
mem_after = process.memory_info().rss / (1024 * 1024)

print(f"Time Used: {end - start:.2f} seconds")
print(f"CPU usage during computation: {cpu_after - cpu_before:.2f}%")
print(f"Memory Used: {mem_after - mem_before:.2f} MB")

cm = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:")
print(cm)

cr = classification_report(y_test, y_pred_classes)
print("Classification Report:")
print(cr)





