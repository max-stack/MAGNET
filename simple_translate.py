from rouge import Rouge

LDA_DEV = "data/test/dev.lda.txt"
NL_DEV = "data/test/dev.nl.txt"
LDA_TRAIN = "data/train/train.lda.txt"
NL_TRAIN = "data/train/train.nl.txt"
OUTPUT = "data/test/output.txt"

def similar_words(lda_dev, lda_train):
    count = 0
    for word in lda_dev.split(" "):
        if word in lda_train:
            count += 1
    
    return count

def addPair(f1, f2):
    for x, y1 in zip(f1, f2):
        yield (x, y1)
    yield (None, None)

output = open(OUTPUT, "w", encoding='utf-8')
for lda_dev in open(LDA_DEV, encoding='utf-8'):
    if lda_dev == None:
        continue
    max_words = 0
    max_line = ""
    for lda_train, nl_train in addPair(open(LDA_TRAIN, encoding='utf-8'), open(NL_TRAIN, encoding='utf-8')):
        if lda_train == None:
            continue
        num_similar_words = similar_words(lda_dev, lda_train)
        if num_similar_words > max_words:
            max_words = num_similar_words
            max_line = nl_train
    output.write(max_line)


tgt = []
out = []

rouge = Rouge()

for tgt_line, out_line in addPair(open(NL_DEV, encoding='utf-8'), open(OUTPUT, encoding='utf-8')):
    if tgt_line is not None and out_line is not None:
        tgt.append(tgt_line)
        out.append(out_line)

print(rouge.get_scores(out, tgt, avg=True))


        