from A0_utils import CsvDataset, Preprocess, PreprocessedWiki
import joblib
import gensim
import gensim.downloader
import json
import os

import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

OUTPUT_FOLDER='input/gensim'

# NOTE TO SELF: Don't use wiki corpus for training. It takes forever to load
# and there are very many words that we don't actually need. Use own dataset
# to train the embedding.

branches = ['amazon.com', 'amazon.fr', 'amazon.es', 'amazon.cn']
for branch in branches:
    train_set = joblib.load(
        'input/train_valid_test/train_{}.joblib'\
            .format(branch.replace('.', '_'))
    )
    model = gensim.models.Word2Vec(sentences=train_set.preprocessed, size=128)
    model.save(
        os.path.join(
            OUTPUT_FOLDER, '{}.model'.format(branch.replace('.', '-'))
        )
    )
    model.wv.save(
        os.path.join(
            OUTPUT_FOLDER, '{}.wv'.format(branch.replace('.', '-'))
        )
    )

# Is it really worth not using a previous corpus?
corpus = gensim.downloader.load('text8')
model = gensim.models.Word2Vec(sentences=corpus, size=300)
model.save(os.path.join(OUTPUT_FOLDER, 'text-8.model'))
model.wv.save(os.path.join(OUTPUT_FOLDER, 'text-8.wv'))

corpus = gensim.downloader.load('text8')
train_set = joblib.load('input/train_valid_test/train_amazon_com.joblib')
model = gensim.models.Word2Vec(size=300)
model.build_vocab(train_set.preprocessed, update=False)
model.build_vocab(corpus, update=True)
total_examples = model.corpus_count + len(train_set)
model.train(corpus, total_examples=total_examples, epochs=5)
model.train(train_set.preprocessed, total_examples=total_examples, epochs=5)
model.save(os.path.join(OUTPUT_FOLDER, 'text8-amazon-com.model'))
model.wv.save(os.path.join(OUTPUT_FOLDER, 'text8-amazon-com.wv'))
