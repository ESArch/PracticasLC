import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords as sw

sw_english = set(sw.words('english'))


def simplified_lesk(word, sentence):
    senses = wn.synsets(word)
    best_sense = senses[0]
    max_overlap = 0
    context = set(nltk.word_tokenize(sentence)).difference(sw_english)

    for sense in senses:

        signature = set(nltk.word_tokenize(sense.definition()))


        for example in sense.examples():
            signature = signature.union(set(nltk.word_tokenize(example)))

        signature = signature.difference(sw_english)

        overlap = len(context.intersection(signature))

        if(overlap > max_overlap):
            max_overlap = overlap
            best_sense = sense

    return best_sense

print(simplified_lesk("bank", "Yesterday I went to the bank to withdraw the money and the credit card did not work").definition())
print(simplified_lesk("book", "If	one	examines	the	words	in	a	book,	one	at	a	time	as	through	\
an	opaque	mask	with	a	hole	in	it	one	word	wide,	then	it	is	\
obviously	impossible	to	determine,	one	at	a	time,	the	meaning	of	the	words").definition())