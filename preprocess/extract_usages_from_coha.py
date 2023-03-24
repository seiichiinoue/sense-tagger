import os, sys
import argparse
import pickle
from tqdm import tqdm
from nltk.tokenize.treebank import TreebankWordDetokenizer

detokenizer = TreebankWordDetokenizer()
pos2id = {'NN': 'n', 'NNS': 'n', 'NNP': 'n', 'NNPS': 'n', 'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v', 'JJ': 'a', 'JJR': 'a', 'JJS': 'a', 'RB': 'r', 'RBR': 'r', 'RBS': 'r'}

def extract_usages(data, target_words, output_path, detokenize=True):
    usages = {tar: [] for tar in target_words}
    for year, line in tqdm(data):
        lemmas = [lemma for _, lemma, _ in line]
        pos = [pos for _, _, pos in line]
        line = [word for word, _, _ in line]
        for tar in target_words:
            if tar in lemmas:
                for i, lemma in enumerate(lemmas):
                    if lemma == tar:
                        target_pos = pos[i]
                        target_pos = pos2id[target_pos]
                        break
                if detokenize:
                    usages[tar].append(str(year)+"\t"+target_pos+"\t"+detokenizer.detokenize(line))
                else:
                    usages[tar].append(str(year)+"\t"+target_pos+"\t"+" ".join(line))
    for tar, usage_list in usages.items():
        with open(os.path.join(output_path, tar+".txt"), "w") as f:
            for line in usage_list:
                f.write(line+"\n")

def load_data(data_path):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    return data

parser = argparse.ArgumentParser()
parser.add_argument('target_words', type=str, nargs='+')
parser.add_argument('--input-path', default='/home/seiichi/coha/data/coha_lemmatized.pickle', type=str)
parser.add_argument('--output-path', default='data/text/coha', type=str)
args = parser.parse_args()

data = load_data(args.input_path)
extract_usages(data=data,
               target_words=args.target_words,
               output_path=args.output_path)