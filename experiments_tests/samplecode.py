from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from torch import nn
import pickle
import json

key_phrase = 'pulls the trigger'

with open('combined_word2id_dict.pkl', 'rb') as f:
    phrase_dict = pickle.load(f)

# phrase_dict = {'pull a trigger': '0', 'squeezes off a quick burst of shots': '1', 'takes aim': '2'}

model = SentenceTransformer('whaleloops/phrase-bert')
# phrase_embs = model.encode( phrase_list )
# [p1, p2, p3] = phrase_embs

p1 = model.encode(key_phrase)

cos_sim = nn.CosineSimilarity(dim=0)

result = {}
for phrase, id in phrase_dict.items():
    print('phrase is:', phrase)
    print('id is:', id)
    emb = model.encode(phrase)
    similarity = cos_sim(torch.tensor(p1), torch.tensor(emb))
    print('similarity is:', similarity)
    print('similarty.item()', similarity.item())
    print('type(similarity.item())', type(similarity.item()))
    result[phrase] = similarity.item()

with open('results_dict.json', 'w') as f:
    json.dump(result, f, indent=4)

with open('results_dict.pkl', 'wb') as f:
    pickle.dump(result, f)

# print(f'The cosine similarity between phrase 1 and 2 is: {cos_sim( torch.tensor(p1), torch.tensor(p2))}')
# print(f'The cosine similarity between phrase 1 and 3 is: {cos_sim( torch.tensor(p1), torch.tensor(p3))}')
# print(f'The cosine similarity between phrase 2 and 3 is: {cos_sim( torch.tensor(p2), torch.tensor(p3))}')
# print(f'The cosine similarity between phrase 4 and 1 is: {cos_sim( torch.tensor(p4), torch.tensor(p1))}')
# print(f'The cosine similarity between phrase 4 and 5 is: {cos_sim( torch.tensor(p4), torch.tensor(p5))}')
