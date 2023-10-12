from genre.fairseq_model import GENRE
from genre.trie import Trie
import pickle
import jsonlines
from tqdm import tqdm
import pprint
import argparse
import os
import json


def main(args):
    # Load the prefix tree (trie)
    with open("data/esco_titles_trie_dict_genre_constrained.pkl", "rb") as f:
        trie = Trie.load_from_dict(pickle.load(f))

    # Load the model
    model_path = f"runs/{args.seed}/{args.model_type}"
    if args.pretrained:
        model_path = "models/fairseq_entity_disambiguation_blink/"
    model = GENRE.from_pretrained(model_path).eval()

    # Load the test set
    with jsonlines.open(args.prediction_path) as reader:
        test_set = list(reader)
    
    output_path = f"results/{str(args.seed)}/{args.model_type}/"
    try:
        os.makedirs(output_path)
    except OSError as e:
        print(e)
    
    with open(f"{output_path}results_predictions_constrained.jsonl", "w") as fout:
        # Initialize counters
        total_examples = 0
        ks = [1, 4, 8, 16, 32]  # You can change 'k' to the desired value

        final_preds = []
        gold = []

        # Evaluate each example in the test set
        for example in tqdm(test_set):
            # if example["output"][0]["answer"] == "UNK":
                # continue
            input_text = example["input"]
            gold.append(example["output"][0]["answer"])
            total_examples += 1

            # Generate predictions for the input text
            predictions = model.sample(
                [input_text],
                beam=32,
                max_len_b=128,
                prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
            )
            if args.write_predict:
                d = {
                    "input": input_text,
                    "preds": [obj["text"] for obj in predictions[0]],
                    "gold": example["output"][0]["answer"],
                }
                fout.write(json.dumps(d))
                fout.write("\n")
            final_preds.append([obj["text"] for obj in predictions[0]])
    
    with open(f"{output_path}results_constrained.out", "a") as fout_results:
        # Calculate accuracy@k
        fout_results.write(model_path)
        fout_results.write("\n")
        for k in ks:
            total_correct_at_k = 0
            for p, g in zip(final_preds, gold):
                top_k_preds = p[:k]
                if g in top_k_preds:
                    total_correct_at_k += 1

            # Calculate the overall accuracy@k for the test set
            overall_accuracy_at_k = total_correct_at_k / total_examples
            fout_results.write(f"Accuracy@{k}: {overall_accuracy_at_k:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--prediction_path", 
                        nargs="?", 
                        type=str, 
                        help="Path to prediction file", 
                        default="datasets/blink_format/valid-esco-kilt.jsonl")
    parser.add_argument("--model_type", 
                        nargs="?", 
                        type=str, 
                        help="name of pre-trained model")
    parser.add_argument("--seed", 
                        nargs="?", 
                        type=int, 
                        help="seed")
    parser.add_argument("--write_predict", 
                        action="store_true", 
                        help="write predictions to stdout")
    parser.add_argument("--pretrained", 
                        action="store_true", 
                        help="use pretrained model instead of fine-tuned one")
    
    args = parser.parse_args()
    main(args)