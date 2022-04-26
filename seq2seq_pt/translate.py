from __future__ import division

import s2s
import torch
import argparse
import math
import time
import logging
from rouge import Rouge
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from rouge_score import rouge_scorer
from nltk.translate import bleu_score, meteor_score
from similarity_score import Similarity_Ratio_English
import nltk
import json
import numpy as np

nltk.download("wordnet")
nltk.download("omw-1.4#")

logging.basicConfig(format='%(asctime)s [%(levelname)s:%(name)s]: %(message)s', level=logging.INFO)
file_handler = logging.FileHandler(time.strftime("%Y%m%d-%H%M%S") + '.log.txt', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
logging.root.addHandler(file_handler)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='translate.py')

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-src', required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-lda', required=True)
parser.add_argument('-tgt',
                    help='True target sequence (optional)')
parser.add_argument('-output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-beam_size', type=int, default=12,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=64,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=100,
                    help='Maximum sentence length.')
parser.add_argument('-replace_unk', action="store_true",
                    help="""Replace the generated UNK tokens with the source
                    token that had the highest attention weight. If phrase_table
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")
parser.add_argument('-verbose', action="store_true",
                    help='logger.info scores and predictions for each sentence')
parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")

parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")
parser.add_argument('-max_lda_words', type=int, default=10)


def reportScore(name, scoreTotal, wordsTotal):
    logger.info("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, scoreTotal / wordsTotal,
        name, math.exp(-scoreTotal / wordsTotal)))


def addone(f):
    for line in f:
        yield line
    yield None


def addPair(f1, f2):
    for x, y1 in zip(f1, f2):
        yield (x, y1)
    yield (None, None)


def addTriple(f1, f2, f3):
    for x, y1, y2 in zip(f1, f2, f3):
        yield (x, y1, y2)
    yield (None, None, None)

def similarity_score(problem1, train_data):
    highest_similarity = 0
    for j in range(len(train_data)):
        test = Similarity_Ratio_English(problem1, train_data[j]["segmented_text"])
        similarity = test.cosine_similarity()
        if similarity > highest_similarity:
            highest_similarity = similarity
    
    return highest_similarity


def main():
    opt = parser.parse_args()
    logger.info(opt)
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    translator = s2s.Translator(opt)

    outF = open(opt.output, 'w', encoding='utf-8')

    predScoreTotal, predWordsTotal, goldScoreTotal, goldWordsTotal = 0, 0, 0, 0

    srcBatch, tgtBatch = [], []
    lda_batch = []

    count = 0
    all_attn = []
    all_topic_attn = []
    all_mix_gate = []

    tgtF = open(opt.tgt, encoding="utf-8") if opt.tgt else None
    for line, lda in addPair(open(opt.src, encoding='utf-8'), open(opt.lda, encoding='utf-8')):

        if (line is not None) and (lda is not None):
            srcTokens = line.strip().split(' ')
            srcBatch += [srcTokens]
            lda_tokens = lda.strip().split(' ')[:opt.max_lda_words]
            lda_batch += [lda_tokens]
            if tgtF:
                tgtTokens = tgtF.readline().split(' ') if tgtF else None
                tgtBatch += [tgtTokens]

            if len(srcBatch) < opt.batch_size:
                continue
        else:
            # at the end of file, check last batch
            if len(srcBatch) == 0:
                break

        predBatch, predScore, goldScore = translator.translate(srcBatch, lda_batch, tgtBatch)

        predScoreTotal += sum(score[0] for score in predScore)
        predWordsTotal += sum(len(x[0]) for x in predBatch)
        # if tgtF is not None:
        #     goldScoreTotal += sum(goldScore)
        #     goldWordsTotal += sum(len(x) for x in tgtBatch)

        rouge = Rouge()
        for b in range(len(predBatch)):
            count += 1
            outF.write(" ".join(predBatch[b][0][0]) + '\n')
            all_attn.append(predBatch[b][0][1].cpu())
            all_topic_attn.append(predBatch[b][0][2].cpu())
            all_mix_gate.append(predBatch[b][0][3].cpu())
            outF.flush()

            if opt.verbose:
                srcSent = ' '.join(srcBatch[b])
                if translator.tgt_dict.lower:
                    srcSent = srcSent.lower()
                logger.info('SENT %d: %s' % (count, srcSent))
                logger.info('PRED %d: %s' % (count, " ".join(predBatch[b][0])))
                logger.info("PRED SCORE: %.4f" % predScore[b][0])

                if tgtF is not None:
                    tgtSent = ' '.join(tgtBatch[b])
                    if translator.tgt_dict.lower:
                        tgtSent = tgtSent.lower()
                    logger.info('GOLD %d: %s ' % (count, tgtSent))
                    # logger.info("GOLD SCORE: %.4f" % goldScore[b])

                if opt.n_best > 1:
                    logger.info('\nBEST HYP:')
                    for n in range(opt.n_best):
                        logger.info("[%.4f] %s" % (predScore[b][n], " ".join(predBatch[b][n])))

                logger.info('')

        srcBatch, tgtBatch = [], []
        lda_batch = []

    reportScore('PRED', predScoreTotal, predWordsTotal)
    # if tgtF:
    #     reportScore('GOLD', goldScoreTotal, goldWordsTotal)

    if tgtF:
        tgtF.close()

    logger.info('{0} copy'.format(translator.copyCount))

    torch.save(all_attn, opt.output + '.attn.pt')
    torch.save(all_topic_attn, opt.output + '.topicattn.pt')
    torch.save(all_mix_gate, opt.output + '.mixgate.pt')

    # tgt = []
    # out = []
    # rouge_scores = []
    # bleu_scores = []
    # meteor_scores = []
    # similarity = []

    # NL_TRAIN = "../data/datasets/mawps_trainset.json"
    # NL_TEST = "../data/datasets/mawps_testset.json"
    # NL_DEV = "../data/datasets/mawps_validset.json"

    # file = open(NL_TRAIN, "r", encoding="utf-8")
    # train_data = json.load(file)

    # file = open(NL_TEST, "r", encoding="utf-8")
    # train_data.extend(json.load(file))

    # file = open(NL_DEV, "r", encoding="utf-8")
    # train_data.extend(json.load(file))


    # i = 0
    # scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    # for tgt_line, out_line in addPair(open(opt.tgt, encoding='utf-8'), open(opt.output, encoding='utf-8')):
    #     if tgt_line is not None and out_line is not None:
    #         tgt.append(tgt_line)
    #         out.append(out_line)
    #         cur_rouge_score = scorer.score(tgt_line, out_line)["rougeL"].fmeasure
    #         cur_bleu_score = bleu_score.sentence_bleu([tgt_line.split()], out_line.split())
    #         cur_meteor_score = meteor_score.meteor_score([tgt_line.split()], out_line.split())

    #         rouge_scores.append(cur_rouge_score)
    #         bleu_scores.append(cur_bleu_score)
    #         meteor_scores.append(cur_meteor_score)
    #         max_ratio = similarity_score(out_line, train_data)
    #         similarity.append(max_ratio)
    #         print(max_ratio)
    #     i += 1

    # sorted_similarity = []

    # sorted_rouge = np.array([x for _,x in sorted(zip(similarity,rouge_scores))])
    # sorted_meteor = np.array([x for _,x in sorted(zip(similarity,meteor_scores))])
    # sorted_bleu = np.array([x for _,x in sorted(zip(similarity,bleu_scores))])
    
    # similarity.sort()
    # similarity = np.array(similarity)

    # x = np.array(list(range(len(rouge_scores))))

    # a_sim, b_sim = np.polyfit(x, similarity, 1)
    # a_rouge, b_rouge = np.polyfit(x, sorted_rouge, 1)
    # a_bleu, b_bleu = np.polyfit(x, sorted_bleu, 1)
    # a_met, b_met = np.polyfit(x, sorted_meteor, 1)

    # plt.scatter(x, sorted_rouge, s=3)
    # plt.scatter(x, sorted_bleu, s=3)
    # plt.scatter(x, sorted_meteor, s=3)
    # plt.scatter(x, similarity, s=3)
    # plt.plot(x, a_rouge*x+b_rouge, label="ROUGE Score")
    # plt.plot(x, a_bleu*x+b_bleu, label="BLEU Score")
    # plt.plot(x, a_met*x+b_met, label="METEOR Score")
    # plt.plot(x, a_sim*x+b_sim, label="Similarity Score")
    # plt.plot(x, sorted_rouge, label="ROUGE Score")
    # plt.plot(x, sorted_bleu, label="BLEU Score")
    # plt.plot(x, sorted_meteor, label="METEOR Score")
    # plt.plot(x, similarity, label="Similarity Score")
    # plt.xlabel("Generated Problem Index")
    # plt.ylabel("Ratio Score")
    # plt.legend()

    #plt.show()

if __name__ == "__main__":
    main()
