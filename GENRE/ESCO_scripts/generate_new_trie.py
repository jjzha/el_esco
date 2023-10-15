# from genre.fairseq_model import GENRE
from genre.fairseq_model import GENRE
from genre.trie import Trie
import pickle
import json



model = GENRE.from_pretrained("runs/276800/bart_large_genre/").eval()

titles = list()

with open("datasets/blink/train.jsonl") as f_train, open("datasets/blink/valid.jsonl") as f_dev, open("datasets/blink/test.jsonl") as f_test:

    # for line in f_train:
    #     data = json.loads(line)
    #     titles.append(data["label_title"])
    
    # for line in f_dev:
    #     data = json.loads(line)
    #     titles.append(data["label_title"])
    
    for line in f_test:
        data = json.loads(line)
        if data["label_title"] == "UNK":
            titles.append("NIL")
        else:
            titles.append(data["label_title"])

# trie = Trie([
    # model.encode(" }} [ {} ]".format(e))[1:].tolist()
    # for e in list(titles) + ["NIL"]
# ])

trie = Trie([2]+model.encode(entity).tolist() for entity in titles).trie_dict
# trie = Trie([
        # model.encode(" {}".format(e))[1:].tolist()
        # for e in titles
    # ]).trie_dict

print(trie)

with open('data/esco_trie_bart_test.pkl', 'wb') as w_f:
    pickle.dump(trie, w_f)
print("finish running!")