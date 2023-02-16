# sense-tagger

OntoNotes sense tagger using PLMs

## Preprocessing

- `preprocess/extract_usages_from_sense_inventories.py`: extracting all usages from OntoNotes. Need to specify path to OntoNotes.
- `preprocess/extract_usages_from_ontonotes.py`: same as below (old version).
- `preprocess/integrate_lemma.py`: integrating lemma (across POS).
- `preprocess/extract_usages_from_coha.py`: extracting all usages from COHA. Need to prepare processed data through [coha](https://github.com/seiichiinoue/coha).

## Sense Tagging

- `sense_tagging.py`: tagging sense using BERT (or other PLMs) and FAISS

