# 入力フレーズに対してコサイン類似度を求めていく
# 類似度の結果をjson, pklで出力

from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from torch import nn
import pickle
import json

# 今回の入力
key_phrase = 'pulls the trigger'
# データセットの読み込み
with open('combined_word2id_dict.pkl', 'rb') as f:
    phrase_dict = pickle.load(f)

# PhraseBERTのモデルの読み込み
model = SentenceTransformer('whaleloops/phrase-bert')

# 入力のベクトル表現を得る len(p1) = 256 の固定長
p1 = model.encode(key_phrase)

cos_sim = nn.CosineSimilarity(dim=0)
result = {}

# データセットの各フレーズに対してiterate
for phrase, id in phrase_dict.items():
    print('phrase is:', phrase)
    print('id is:', id)
    # フレーズのベクトル表現を得る
    emb = model.encode(phrase)
    # 入力とフレーズとのコサイン類似度を求める
    similarity = cos_sim(torch.tensor(p1), torch.tensor(emb))
    print('similarity is:', similarity)
    # print('similarty.item()', similarity.item())
    result[phrase] = similarity.item()


# 結果の保存
with open('results_dict.json', 'w') as f:
    json.dump(result, f, indent=4)

with open('results_dict.pkl', 'wb') as f:
    pickle.dump(result, f)

# print(f'The cosine similarity between phrase 1 and 2 is: {cos_sim( torch.tensor(p1), torch.tensor(p2))}')
# print(f'The cosine similarity between phrase 1 and 3 is: {cos_sim( torch.tensor(p1), torch.tensor(p3))}')
# print(f'The cosine similarity between phrase 2 and 3 is: {cos_sim( torch.tensor(p2), torch.tensor(p3))}')
# print(f'The cosine similarity between phrase 4 and 1 is: {cos_sim( torch.tensor(p4), torch.tensor(p1))}')
# print(f'The cosine similarity between phrase 4 and 5 is: {cos_sim( torch.tensor(p4), torch.tensor(p5))}')
