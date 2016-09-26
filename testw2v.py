from gensim.models import Word2Vec, Doc2Vec, Phrases

sentences = ['Note that there is a gensim.models.phrases module which lets', 'actually multiword expressions,']
sentences_list = [[u'test', u'this', u'one', u'try', u'this'],
                  [u'too', u'test', u'this', u'one', u'try', u'this', u'too', u'wppppp']]

# print(sentences_list[0])

model = Phrases(sentences_list)
# model.build_vocab(sentences=sentences_list[0])
# print(model.vocab)

w2v = Word2Vec(iter=1, min_count=1)
w2v.build_vocab(sentences_list)
# for word in w2v.vocab:
#     print(word)

print(w2v.most_similar("wppppp"))
print(w2v.similarity("test", "this"))
