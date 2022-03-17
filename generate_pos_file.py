import json
import nltk
from nltk.data import load
from sklearn.preprocessing import LabelBinarizer
import torch

nltk.download('averaged_perceptron_tagger')

with open("data/word2idx.json", "r") as file:
  words_file = file.read()
words_dic = json.loads(words_file)
words = list(words_dic.keys())
tagged = nltk.pos_tag(words)

pos_tags_list = list(set([tagged[i][1] for i in range(len(tagged))]))

encoder = LabelBinarizer()
transformed_label = encoder.fit_transform(pos_tags_list)
labels = encoder.classes_
mappings = {}
for index, label in zip(range(len(labels)), labels):
  mappings[label]=index

data = torch.tensor([transformed_label[mappings[tagged[j][1]]] for j in range(len(tagged))])

def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj, fh)