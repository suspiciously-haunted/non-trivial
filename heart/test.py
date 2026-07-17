import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def average_percentage_difference(recon, avg, epsilon=1):
    recon = np.array(recon)
    avg = np.array(avg)

    abs_diff = np.abs(recon - avg)

    denom = np.maximum(np.abs((avg+recon)//2), epsilon)

    # If both recon and avg are zero, set denom to 1 temporarily to avoid div by zero
    # Then we will mask those cases to zero later
    zero_mask = (recon == 0) & (avg == 0)

    perc_diff = abs_diff / denom * 100

    # For positions where both are zero, override perc_diff to 0
    perc_diff = np.where(zero_mask, 0.0, perc_diff)

    avg_perc_diff = np.mean(perc_diff)
    return avg_perc_diff

examples = [
    # 1
    ([ 55, 1, 165, 78, 131, 90, 1, 1, 0, 0, 1],
[ 55, 1, 165, 78, 130, 90, 1, 1, 0, 0, 1]),

    # 2
    ([ 56, 1, 167, 79, 128, 84, 1, 1, 0, 0, 1],
[ 54, 2, 168, 78, 125, 82, 1, 1, 0, 0, 1]),

    # 3
    ([ 55, 2, 168, 72, 120, 81, 1, 1, 0, 0, 1],
[ 52, 2, 168, 74, 124, 81, 1, 1, 0, 0, 1]),
]

for i, (recon, avg) in enumerate(examples, 1):
    score = average_percentage_difference(recon, avg)
    print(f"Example {i} Average Percentage Difference: {score:.2f}%")