from difflib import SequenceMatcher

EQU_DATA_DEV = "data/test/dev.equ.txt"
NL_DATA_DEV = "data/test/dev.nl.txt"
LDA_DATA_DEV = "data/test/dev.lda.txt"

NL_TRAIN = "data/train/train.nl.txt"


equ_file = open(EQU_DATA_DEV, "r", encoding="utf-8")
nl_file = open(NL_DATA_DEV, "r", encoding="utf-8")
lda_file = open(LDA_DATA_DEV, "r", encoding="utf-8")

equ_lines = equ_file.readlines()
nl_lines = nl_file.readlines()
lda_lines = lda_file.readlines()

equ_file = open(EQU_DATA_DEV, "w", encoding="utf-8")
nl_file = open(NL_DATA_DEV, "w", encoding="utf-8")
lda_file = open(LDA_DATA_DEV, "w", encoding="utf-8")

for i in range(len(nl_lines)):
    max_ratio = 0
    for nl_train in open(NL_TRAIN, "r", encoding="utf-8"):
        max_ratio = max(max_ratio, SequenceMatcher(None, nl_lines[i], nl_train).ratio())
    print(max_ratio)
    if(max_ratio < 0.75):
        equ_file.write(equ_lines[i])
        nl_file.write(nl_lines[i])
        lda_file.write(lda_lines[i])


