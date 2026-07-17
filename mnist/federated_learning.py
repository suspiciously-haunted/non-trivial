import numpy as np
import tensorflow as tf
from client import train_client_model, train_malicious_model, train_tenseal_model, decrypt_weights, create_context
from aggregator import robust_aggregate, gradient_clipping, aggregate_models, aggregate_encrypted

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def federated_learning(num_rounds, num_clients, num_malicious, num_epochs):
    model = create_model()  
    context = create_context()
    original_shapes = None

    for round_num in range(num_rounds):
        print(f"Round {round_num + 1}/{num_rounds}")
        
        all_client_weights = []

        for _ in range(num_clients):
            client_weights, shapes = train_tenseal_model(model, num_epochs, context)
            all_client_weights.append(client_weights)
            if original_shapes is None:
                original_shapes = shapes
        # for _ in range(num_clients):
        #     client_weights = train_client_model(model, num_epochs)
        #     all_client_weights.append(client_weights)
        for _ in range(num_malicious//2):
            malicious_weights = train_malicious_model(model, num_epochs, mode="malicious")
            all_client_weights.append(malicious_weights)
        for _ in range(num_malicious - num_malicious//2):
            malicious_weights = train_malicious_model(model, num_epochs, mode="random")
            all_client_weights.append(malicious_weights)

        # clipped_weights = [gradient_clipping(weight) for weight in all_client_weights]
        # global_weights = robust_aggregate(clipped_weights)

        global_weights = aggregate_encrypted(all_client_weights)
        global_weights = decrypt_weights(global_weights, original_shapes)
        
        model.set_weights(global_weights)
        print(f"Round {round_num + 1} complete.")
    
    model.save('test.keras')
    return model


