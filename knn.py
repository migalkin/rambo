import torch
from gensim.models import KeyedVectors as kv

# Loading model's rel emb and ent emb matrix
MODEL_PATH = 'data/models/fb15k237/TransE/0/model.torch'
model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
ent_w, rel_w = model['entity_embeddings.weight'], \
               model['relation_embeddings.weight']

# Creating dictionary.
ent, rel = {}, {}
ent_embsize, rel_embsize = 0, 0
for i, weight in enumerate(ent_w):
    ent[i] = weight.numpy()
    ent_embsize = len(weight)
for i, weight in enumerate(rel_w):
    rel[i] = weight.numpy()
    rel_embsize = len(weight)


def create_file(emb: dict, emb_size, file_name):
    final_strings = []
    final_strings.append([str(len(emb) - 1)] + [str(emb_size)] + ['\n'])

    for key, value in emb.items():
        if key != 0:
            final_strings.append([str(key)] +
                                 [str(v) for v in value] + ['\n'])

    with open(file_name, 'a') as file:
        for f in final_strings:
            file.write(" ".join(f))


# Storing files on disk in oder for Gensim to read.
create_file(rel, rel_embsize, './data/rel.txt')
create_file(ent, ent_embsize, './data/ent.txt')

# Most similar words
rel_model = kv.load_word2vec_format('./data/rel.txt')
print(rel_model.similar_by_word("1"))
