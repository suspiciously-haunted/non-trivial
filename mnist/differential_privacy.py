import tensorflow as tf
import numpy as np

def apply_differential_privacy(gradients, noise_multiplier=1.0):
    """
    Adds Gaussian noise to gradients to protect against gradient inversion attacks.
    """
    noise = np.random.normal(0, noise_multiplier, size=gradients.shape)
    noisy_gradients = gradients + noise
    return noisy_gradients

def train_with_dp(model, x_train, y_train, epochs=1, batch_size=32, noise_multiplier=1.0):
    # Train the model with added differential privacy
    for epoch in range(epochs):
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            
            with tf.GradientTape() as tape:
                predictions = model(x_batch)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch, predictions)
                
            gradients = tape.gradient(loss, model.trainable_variables)
            # Apply differential privacy to the gradients
            noisy_gradients = [apply_differential_privacy(grad, noise_multiplier) for grad in gradients]
            
            # Update the model weights using noisy gradients
            optimizer = tf.keras.optimizers.Adam()
            optimizer.apply_gradients(zip(noisy_gradients, model.trainable_variables))

# Example of using differential privacy during training
model = create_model()
train_with_dp(model, x_train, y_train, epochs=5, noise_multiplier=1.0)
