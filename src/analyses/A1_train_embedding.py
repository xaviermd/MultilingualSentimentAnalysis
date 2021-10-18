import logging
import gensim
import gensim.downloader

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

# corpus = gensim.downloader.load('text8')
corpus = gensim.downloader.load('wiki-english-20171001')

model = gensim.models.Word2Vec(corpus)

model.save('/usr/local/data/xaviermd/analyses/wiki-english.model')
