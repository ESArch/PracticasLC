from nltk.corpus import cess_esp
from nltk.tag import hmm
from nltk.tag import tnt
from random import shuffle

def reduce(full_sents):
    red_sents = []

    def reduce_tag(tag):
        if tag[0] == "v" or tag[0] == "F":
            return tag[:3]
        else:
            return tag[:2]

    for sent in full_sents:
        red_sent = []
        for w,t in sent:
            if w != "*0*":
                red_sent+=[(w,reduce_tag(t))]
        red_sents += [red_sent]

    return red_sents

def evaluateTnT():
    tagger = tnt.TnT()
    train = int(len(fsents)*0.9)
    tagger.train(rsents[:train])
    precisionTnT = tagger.evaluate(rsents[train:])
    print(precisionTnT)

def evaluateHMM():
    train = int(len(fsents) * 0.9)
    tagger = hmm.HiddenMarkovModelTagger.train(rsents[:train])
    precisionHMM = tagger.evaluate(rsents[train:])
    print(precisionHMM)


def crossValidationTnT():
    numSents = len(rsents)
    precision = 0.0
    for i in range(10):
        p1 = int(i*numSents/10)
        p2 = int((i+1)*numSents/10)
        tagger = tnt.TnT()
        tagger.train(rsents[:p1]+rsents[p2:])
        precisionK = tagger.evaluate(rsents[p1:p2])
        print("Precision con particion {} = {}".format(i+1, precisionK))
        precision += precisionK
    print(precision/10)

def crossValidationHMM():
    numSents = len(rsents)
    precision = 0.0
    for i in range(10):
        p1 = int(i*numSents/10)
        p2 = int((i+1)*numSents/10)
        tagger = hmm.HiddenMarkovModelTagger.train(rsents[:p1]+rsents[p2:])
        precisionK = tagger.evaluate(rsents[p1:p2])
        print("Precision con particion {} = {}".format(i+1, precisionK))
        precision += precisionK
    print(precision/10)

fsents = cess_esp.tagged_sents()
rsents = reduce(fsents)
shuffle(rsents)


evaluateTnT()
evaluateHMM()



crossValidationTnT()
crossValidationHMM()


