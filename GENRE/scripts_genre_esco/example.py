# from genre.hf_model import GENRE
from genre.fairseq_model import GENRE
from genre.trie import Trie
import pickle

# load the prefix tree (trie)
with open("data/esco_titles_trie_dict.pkl", "rb") as f:
    trie = Trie.load_from_dict(pickle.load(f))

model = GENRE.from_pretrained("models/esco_genre_bart_base/").eval()

sentences = ["You need knowledge of [START_END] Python [END_ENT]."]

print(model.sample(
    sentences,
    beam=3,
    max_len_b=15,
    prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()))),