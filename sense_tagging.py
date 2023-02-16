import os
import json
import pickle
import faiss
import numpy as np
from nltk.stem import WordNetLemmatizer
from transformers import BertConfig, BertForPreTraining, \
                         BertJapaneseTokenizer, BertModel, AutoTokenizer, \
                         BertTokenizer

model_name = "bert-base-uncased"  # recommended
json_path = "data/ontonotes/usages_integrated.json"
data_path = "data/coha"
lemmatizer = WordNetLemmatizer()


def from_pretrained(model_name):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    return tokenizer, model

"""copied from kazumakobayashi/Semantic_change_2022"""
def find_target_token_spans(tokens, target_word):
    tmp = ""
    spans = []
    s, e = 0, 0
    for i in range(len(tokens)):
        if target_word == tokens[i]:
            spans.append((i, i))
            tmp = ""
            continue
        if target_word.startswith(tokens[i]):
            s = i
            e = i
            tmp = tokens[i]
            continue
        tmp += tokens[i]
        if target_word == tmp:
            spans.append((s,e+1))
            tmp = ""
        if target_word.startswith(tmp):
            e = i
    return spans
    
def get_ontonotes_embeddings(tokenizer, model, data, target_word):
    cnt_all, cnt_err = 0, 0
    embs_all = {}
    for pos_sense, definition_sentences in data.items():
        pos, sense = pos_sense.split("-")
        definition, sentences = definition_sentences
        embs_k = []
        for sent in sentences:
            """
            raw sentence: 'Somerwil coached the rowing program there for seven years'
            tokenized: ['some', '##r', '##wil', 'coached', 'the', 'rowing', 'program', 'there', 'for', 'seven', 'years']
            target_position: 3
            """
            tokens = tokenizer.tokenize(sent)
            target_position = -1
            for i, w in enumerate(tokens):
                if "#" in w:
                    continue
                lemma = lemmatizer.lemmatize(w.lower(), pos)
                if lemma == target_word:
                    target_position = i
            if target_position == -1:  # cannot find target_word in sentence
                cnt_err += 1
                continue
            emb = model(**tokenizer(sent, return_tensors="pt")).last_hidden_state[0]
            target_emb = emb[target_position + 1].to('cpu').detach().numpy().tolist()
            embs_k.append(target_emb)
        embs_all[pos_sense] = embs_k
    print("[Info] error counts in OntoNotes data:", cnt_err)
    return embs_all

def get_coha_embeddings(tokenizer, model, data, target_word):
    embs = []
    cnt_err = 0
    for year, pos, sent in data:
        tokens = tokenizer.tokenize(sent)
        target_position = -1
        for i, w in enumerate(tokens):
            if "#" in w:
                continue
            lemma = lemmatizer.lemmatize(w.lower(), pos)
            if lemma == target_word:
                target_position = i
        if target_position == -1:
            cnt_err += 1
            continue
        emb = model(**tokenizer(sent, return_tensors="pt")).last_hidden_state[0]
        target_emb = emb[target_position + 1].to('cpu').detach().numpy().tolist()
        embs.append([sent, target_emb])
    print("[Info] error counts in COHA data:", cnt_err)
    return embs

def nearest_neighbor_search(ontonotes_data, coha_data, k=1):
    ontonotes_embs, ontonotes_labels = [], []
    for sense_label, embs in ontonotes_data.items():
        for emb in embs:
            ontonotes_embs.append(emb)
            ontonotes_labels.append(sense_label)
    coha_embs, coha_sents = [], []
    for sent, emb in coha_data:
        coha_sents.append(sent)
        coha_embs.append(emb)
    assert len(ontonotes_embs[0]) == len(coha_embs[0])
    emb_dim = len(ontonotes_embs[0])
    index = faiss.IndexFlatL2(emb_dim)
    index.add(np.array(ontonotes_embs).astype('float32'))
    results = []
    for i, usage_emb in enumerate(coha_embs):
        dist, ind = index.search(np.array([usage_emb]).astype('float32'), k)
        results.append([dist[0][0], ind[0][0], ontonotes_labels[ind[0][0]], coha_sents[i]])
    return results

def execute_faiss(target_word):
    with open(f"data/embeddings/{target_word}_ontonotes.json", "r") as f:
        ontonotes_embeddings = json.load(f)
    with open(f"data/embeddings/{target_word}_coha.json", "r") as f:
        coha_embeddings = json.load(f)
    results = nearest_neighbor_search(ontonotes_embeddings, coha_embeddings)
    used_senses = set()
    for _, _, label, _ in results:
        used_senses.add(label)
    print(len(used_senses), used_senses)
        
def load_ontonotes_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def load_coha_data(data_path, target_word):
    with open(os.path.join(data_path, target_word + ".txt"), "r") as f:
        data = [line.strip().split("\t") for line in f.readlines()]
    return data

def main(target_words):
    tokenizer, model = from_pretrained(model_name)
    ontonotes_data = load_ontonotes_data("data/ontonotes/usages_integrated.json")
    for target_word in target_words:
        ontonotes_data_path = f"data/embeddings/{target_word}_ontonotes.json"
        coha_data_path = f"data/embeddings/{target_word}_coha.json"
        if os.path.exists(ontonotes_data_path) and os.path.exists(coha_data_path):
            print("Data already exisits!")
            execute_faiss(target_word)
            continue
        assert target_word in ontonotes_data.keys()
        assert os.path.exists(os.path.join(data_path, target_word + ".txt"))
        target_coha_data = load_coha_data(data_path, target_word)
        target_ontonotes_data = ontonotes_data[target_word]
        ontonotes_embeddings = get_ontonotes_embeddings(tokenizer, model, target_ontonotes_data, target_word)
        coha_embeddings = get_coha_embeddings(tokenizer, model, target_coha_data, target_word)
        with open(ontonotes_data_path, "w") as f:
            json.dump(ontonotes_embeddings, f, indent=4)
        with open(coha_data_path, "w") as f:
            json.dump(coha_embeddings, f, indent=4)
        execute_faiss(target_word)
        
if __name__ == "__main__":
    main(["coach"])