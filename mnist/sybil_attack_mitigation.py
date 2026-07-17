import numpy as np
from model import create_model
from client import train_client_model, train_malicious_model
from copy import deepcopy

class Client:
    def __init__(self, id, is_legitimate=True):
        self.id = id
        self.is_legitimate = is_legitimate
        self.reputation = 1.0

    def send_weights(self, global_model, num_epochs):
        local_model = deepcopy(global_model)
        if self.is_legitimate:
            weights = train_client_model(local_model, num_epochs)
        else:
            weights = train_malicious_model(local_model, num_epochs, mode="random") 
        return weights

    def update_reputation(self, reference_weights, client_weights, threshold=0.3):
        """
        Decrease reputation if deviation from reference (global) weights is too large.
        """
        diffs = [np.linalg.norm(c - r) for c, r in zip(client_weights, reference_weights)]
        avg_diff = np.mean(diffs)

        if avg_diff < threshold:
            self.reputation = min(1.0, self.reputation + 0.05)
        else:
            self.reputation = max(0.0, self.reputation - 0.1)

def aggregate_with_reputation(client_weights, client_reputations):
    """
    Weighted average of client weights based on their reputation.
    """
    stacked_weights = np.array(client_weights)
    normalized_reputations = np.array(client_reputations) / np.sum(client_reputations)
    weighted_average = np.tensordot(normalized_reputations, stacked_weights, axes=([0], [0]))
    return weighted_average

# Federated learning with reputation-based aggregation
def federated_learning_with_reputation(num_rounds=5, num_clients=10, num_epochs=1, malicious_ids=[3]):
    clients = [Client(id=i, is_legitimate=(i not in malicious_ids)) for i in range(num_clients)]
    model = create_model()

    for rnd in range(num_rounds):
        print(f"Round {rnd + 1}/{num_rounds}")

        client_weights = []
        for client in clients:
            weights = client.send_weights(model, num_epochs)
            client_weights.append(weights)

        reputations = [client.reputation for client in clients]
        global_weights = aggregate_with_reputation(client_weights, reputations)

        model.set_weights(global_weights)

        # Update reputations
        for client, weights in zip(clients, client_weights):
            client.update_reputation(global_weights, weights)

        print(f"Reputations: {[round(c.reputation, 2) for c in clients]}")

    return model

model = federated_learning_with_reputation(
    num_rounds=10,
    num_clients=10,
    num_epochs=1,
    malicious_ids=[3, 7]
)
