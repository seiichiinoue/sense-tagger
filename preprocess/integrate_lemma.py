import json

usages_path = "data/text/ontonotes/usages.json"
output_path = "data/text/ontonotes/usages_integrated.json"

with open(usages_path, "r") as f:
    usages = json.load(f)

usages_new = {}
for lemma_pos, vals in usages.items():
    if lemma_pos == "noun_grouping_template":
        continue
    if len(lemma_pos.split("-")) > 2:  # in the case of phrase like `short-circuit-v`
        continue
    lemma, pos = lemma_pos.split("-")
    if lemma not in usages_new.keys():
        usages_new[lemma] = {}
    for sense_num, definition_usages in vals.items():
        pos_sense_num = pos + "-" + sense_num
        usages_new[lemma][pos_sense_num] = definition_usages

with open(output_path, "w") as f:
    json.dump(usages_new, f, indent=4)