from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from torch import nn
import pickle
import json

# with open('combined_word2id_dict.pkl', 'rb') as f:
#     data = pickle.load(f)
#
# print(data)

# 基準となるフレーズ
phrase_1 = 'pulls the trigger'
#
data = {'pull a trigger': '0', 'squeezes off a quick burst of shots': '1', 'takes aim': '2', 'picks up his gun': '3', 'squeezes off a quick burst of shots': '4'}
#
model = SentenceTransformer('whaleloops/phrase-bert')
p1 = model.encode( phrase_1 )


# count = 0
# for phrase, embedding in zip(phrase_list, phrase_embs):
#     count += 1
#     print(f"Phrase {count}:", phrase)
    # print("Embedding:", embedding)
    # print("")

cos_sim = nn.CosineSimilarity(dim=0)

# print(f'The cosine similarity between phrase 1 and 2 is: {cos_sim( torch.tensor(p1), torch.tensor(p2))}')
# print(f'The cosine similarity between phrase 1 and 3 is: {cos_sim( torch.tensor(p1), torch.tensor(p3))}')
# print(f'The cosine similarity between phrase 2 and 3 is: {cos_sim( torch.tensor(p2), torch.tensor(p3))}')

# results = {}
for word, id in data.items():
    print('sentence is:', word)
    print('id is:', id)
    emb = model.encode(word)
    similarity = cos_sim( torch.tensor(p1), torch.tensor(emb))
    print('similarity between p1 and this phrase is:', similarity)
#
# print(results)
# with open('results_dict.json', 'w') as f:
#     json.dump(results, f, indent=4)
#
# with open('results_dict.pkl', 'wb') as f:
#     pickle.dump(results, f)
