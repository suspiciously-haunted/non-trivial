from tensorflow.keras.datasets import mnist
import numpy as np
import tenseal as ts

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train.reshape(-1, 784), y_train, x_test.reshape(-1, 784), y_train

def train_client_model(model, num_epochs):
    x_train, y_train, x_test, y_test = load_data()
    model.fit(x_train, y_train, epochs=num_epochs, batch_size=32, verbose=0)
    
    return model.get_weights()

def train_malicious_model(model, num_epochs, mode):
    if mode == "random":
        random_weights = []
        for layer_weights in model.get_weights():
            random_layer = np.random.normal(loc=0.0, scale=1.0, size=layer_weights.shape)
            random_weights.append(random_layer)
        return random_weights

    elif mode == "malicious":
        # Standard label-poisoning attack
        x_train, y_train, x_test, y_test = load_data()

        poisoned_x = x_train[y_train == 3]
        poisoned_y = np.full((poisoned_x.shape[0],), 5)  # Flip labels from 3 → 5

        model.fit(poisoned_x, poisoned_y, epochs=num_epochs, batch_size=32, verbose=0)

        return model.get_weights()
    
    else:
        raise ValueError("Mode must be 'random' or 'malicious'")


def train_tenseal_model(model, num_epochs, context):
    x_train, y_train, x_test, y_test = load_data()
    model.fit(x_train, y_train, epochs=num_epochs, batch_size=32, verbose=0)

    encrypted_weights, original_shapes = encrypt_model(model, context)
    return encrypted_weights, original_shapes


def encrypt_model(model, context):
    encrypted_weights = []
    for weight in model.get_weights():
        flattened = np.array(weight).flatten().tolist()
        encrypted_weight = ts.ckks_vector(context, flattened)
        encrypted_weights.append(encrypted_weight)
    original_shapes = [w.shape for w in model.get_weights()]
    return encrypted_weights, original_shapes

def decrypt_weights(encrypted_weights, original_shapes):
    decrypted_weights = []
    for enc_weight, shape in zip(encrypted_weights, original_shapes):
        flat_weight = enc_weight.decrypt()
        weight = np.array(flat_weight).reshape(shape)
        decrypted_weights.append(weight)
    return decrypted_weights

def create_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=32768,
        coeff_mod_bit_sizes=[60, 40, 60]
    )
    context.generate_galois_keys()
    context.global_scale = 2**40
    return context
