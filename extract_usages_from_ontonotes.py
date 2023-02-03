import os

# the location of annotation data
# in this directory, there are some subdirectory: `bc  bn  mz  nw  pt  tc  wb`
base_dir = "/cldata/LDC/ontonotes-release-5.0/data/files/data/english/annotations"

def onf_parser(texts):
    def _process_line(text):
        text = text.strip()
        text = text.replace(".", " .").replace(",", " ,").replace("?", " ?").replace("!", " !").replace("-", " - ")
        return text
    
    i = 0
    sents = []
    while i < len(texts):
        if texts[i].startswith("Plain sentence:"):
            tmp = ""
            i += 2
            while True:
                tmp += _process_line(texts[i])
                i += 1
                if texts[i].startswith("Treebanked sentence:"):
                    break
                else:
                    tmp += " "
            sents.append(tmp.split())
        else:
            i += 1
    return sents

def extract_usages(target_words):
    target_words = set(target_words)
    word_usages = {tar:[] for tar in target_words}
    cnt = 0
    for sub_dir in os.listdir(base_dir):  # `bc  bn  mz  nw  pt  tc  wb`
        for subsub_dir in os.listdir(os.path.join(base_dir, sub_dir)):  # `cctv  cnn  map.txt  ...`
            if os.path.isfile(os.path.join(base_dir, sub_dir, subsub_dir)):
                continue
            for subsubsub_dir in os.listdir(os.path.join(base_dir, sub_dir, subsub_dir)):  # `00  01  ...`
                tar_files = os.listdir(os.path.join(base_dir, sub_dir, subsub_dir, subsubsub_dir))
                tar_files_prefix = set([".".join(tar.split(".")[:-1]) for tar in tar_files])
                for tar_prefix in tar_files_prefix:
                    # if there are no sense, continue
                    if not os.path.exists(os.path.join(base_dir, sub_dir, subsub_dir, subsubsub_dir, tar_prefix+".sense")):
                        continue
                    texts = open(os.path.join(base_dir, sub_dir, subsub_dir, subsubsub_dir, tar_prefix+".onf"), "r").readlines()
                    senses = open(os.path.join(base_dir, sub_dir, subsub_dir, subsubsub_dir, tar_prefix+".sense"), "r").readlines()
                    sents = onf_parser(texts)  # list of sentence (list)
                    for i, sense in enumerate(senses):
                        sense = sense.split()
                        if len(sense) == 6:  # not inter-annotator agreement
                            info, sent_num, word_num, lemma_pos, q, sense_num = sense
                            if len(lemma_pos.split("-")) > 2:  # hyphen-separated word like "short-circuit-v"
                                continue
                            lemma, pos = lemma_pos.split("-")                            
                            continue
                        elif len(sense) == 5:  # inter-annotator agreement
                            info, sent_num, word_num, lemma_pos, sense_num = sense
                            if len(lemma_pos.split("-")) > 2:  # hyphen-separated word like "short-circuit-v"
                                continue
                            lemma, pos = lemma_pos.split("-")
                        else:
                            print("Parsing error")
                            exit()
                        if lemma == "coach":
                            cnt += 1
                        if lemma not in target_words:
                            continue
                        sent = sents[int(sent_num)]
                        if int(word_num) >= len(sent):
                            continue
                        word = sents[int(sent_num)][int(word_num)]
                        word_usages[lemma].append([" ".join(sent), word, lemma, pos, sense_num])
    print(cnt)
    return word_usages

if __name__ == "__main__":
    word_usages = extract_usages(["crash"])
    print(word_usages)