import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
from tensorflow.keras.models import load_model
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy_statement

(ds_train, ds_test), ds_info = tfds.load(
    'mnist', split=['train', 'test'], shuffle_files=True,
    as_supervised=True, with_info=True
)

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return tf.reshape(image, [-1]), label

noise_multiplier = 1.1
l2_norm_clip = 1.0
batch_size = 256
epochs = 10
microbatches = batch_size 

ds_train = ds_train.map(preprocess).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

optimizer = DPKerasAdamOptimizer(
    l2_norm_clip=1.0,
    noise_multiplier=noise_multiplier,
    num_microbatches=microbatches,
    learning_rate=0.001
)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction=tf.keras.losses.Reduction.NONE
)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
model.fit(ds_train, epochs=epochs)
model.save("model_dpsgd.keras")

# model = tf.keras.models.load_model(
#     "model_dpsgd.keras",
#     custom_objects={'DPKerasAdamOptimizer': DPKerasAdamOptimizer}
# )

statement = compute_dp_sgd_privacy_statement(
    number_of_examples=60000,
    batch_size=batch_size,
    num_epochs=epochs,
    noise_multiplier=noise_multiplier,
    delta=1e-5
)
print(statement)
