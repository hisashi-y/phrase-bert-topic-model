import json
from operator import itemgetter

with open('results_dict.json', 'r') as f:
    data = json.load(f)

k = 5

k_nearest_neighbors = sorted(data.items(), key=itemgetter(1), reverse=True)[:k]
print(k_nearest_neighbors)
