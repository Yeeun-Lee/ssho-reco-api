import pandas as pd
import numpy as np
import io

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import KeyedVectors

def tokenize(sen):
    return sen.lower().split()

class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

if __name__=="__main__":
    train_docs = pd.read_csv("../assets/translation_210106.csv")[['id', 'translated']]
    # tokenized = [tokenize(sen) for sen in train_docs.translated.values]
    print(train_docs.head().values)
    sentences = [TaggedDocument(words = value[1], tags=[str(value[0])]) for value in train_docs.values]


    epoch_logger = EpochLogger()

    model = Doc2Vec(alpha=0.025, min_alpha=0.025, epochs = 30,
                    size = 3)
    model.build_vocab(sentences)

    model.train(sentences, total_examples=model.corpus_total_words,
                epochs = model.epochs)

    sample = "Fleece trousers"
    sample_vector = model.infer_vector(sample.lower().split())
    vecs = model.docvecs.most_similar(positive = [sample_vector])
    rank = 1
    for vec in vecs:
        print("Rank : ", rank)
        print(train_docs[train_docs['id'] == int(vec[0])]['translated'])
        rank+=1
        print()

    model.save("d2v_title.mod")
    out_v = io.open("d2v_title.tsv", "w", encoding = "utf-8")
    out_m = io.open("title_meta.tsv", "w", encoding = "utf-8")

    for i in range(len(train_docs)):
        word = '_'.join(map(str, train_docs[['id', 'translated']].values[i]))
        vec = model.docvecs.doctag_syn0[i]
        out_m.write(word+"\n")
        out_v.write("\t".join([str(x) for x in vec])+"\n")
    out_v.close()
    out_m.close()
