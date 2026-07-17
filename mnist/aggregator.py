
import numpy as np

def gradient_clipping(weights, clip_value=1.0):
    return [np.clip(weight, -clip_value, clip_value) for weight in weights]

def robust_aggregate(client_weights):
    median_weights = []
    for weights_list in zip(*client_weights):
        median_weights.append(np.median(np.array(weights_list), axis=0))
    return median_weights

def aggregate_models(models_weights):
    avg_weights = []
    for weights_list in zip(*models_weights):
        avg_weights.append(np.mean(np.array(weights_list), axis=0))
    return avg_weights

def aggregate_encrypted(all_client_weights):
    num_clients = len(all_client_weights)
    num_weights = len(all_client_weights[0])
    
    aggregated_weights = []
    for i in range(num_weights):
        agg_weight = all_client_weights[0][i]
        for c in range(1, num_clients):
            agg_weight = agg_weight + all_client_weights[c][i]
        agg_weight = agg_weight * (1.0 / num_clients) 
        aggregated_weights.append(agg_weight)
    return aggregated_weights