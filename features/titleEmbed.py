from datetime import datetime
import json
import io

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec

from features.translation import finalItems


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

class TitleEmbeding():
    def __init__(self, file = None, embed_size = 3, epochs = 30, alpha = 0.025, min_alpha = 0.025, verbose = False):
        self.embed_size = embed_size
        self.items = finalItems(file)

        # model parameters
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.epochs = epochs
        self.verbose = verbose

        self.model = Doc2Vec(alpha = self.alpha, min_alpha= self.min_alpha, epochs = self.epochs)

        self.date = datetime.now().strftime("%y%m%d")

    def build_vocabulary(self):
        self.sentences = [TaggedDocument(words=value['translated'],
                                          tags=[str(value['id'])]) for value in self.items]
        self.model.build_vocab(self.sentences)

    def train(self):
        self.build_vocabulary()
        callbacks = []
        if self.verbose:
            callbacks.append(EpochLogger())

        self.model.train(self.sentences, total_examples=self.model.corpus_total_words,
                         epochs = self.model.epochs,)
    def save(self):
        self.model.save("d2v_{}.mod".format(self.date))

    def save_tsv(self):
        out_v = io.open("d2v_title.tsv", "w", encoding="utf-8")
        out_m = io.open("title_meta.tsv", "w", encoding="utf-8")

        for i in range(len(self.items)):
            word = ':'.join([str(self.items[i]['id']), self.items[i]['title'], self.items[i]['mallNm']])
            vec = self.model.docvecs.doctag_syn0[i]
            out_m.write(word + "\n")
            out_v.write("\t".join([str(x) for x in vec]) + "\n")
        out_v.close()
        out_m.close()

    def get_vector(self, file = None):
        vectors = self.model.docvecs.doctag_syn0
        for i in range(len(self.items)):
            self.items[i]['title_vector'] = vectors[i]
        if file!=None:
            with open(file, "w") as f:
                json.dump(self.items, f)
        else:
            return self.items


if __name__=="__main__":
    vectorizer = TitleEmbeding()
    vectorizer.train() # 학습
    # model save
    vectorizer.save()

    # embedding vector가 추가된 json파일
    items = vectorizer.get_vector(file = None) # file : 파일명, 입력 안할 시 파일 저장 x, return됨

    # tsv파일 저장(tensorboard embedding 에 시각화 하기 위한 파일)
    vectorizer.save_tsv()