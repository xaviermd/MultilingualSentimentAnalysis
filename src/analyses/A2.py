from A0_utils import CsvDataset, SentimentLSTM, TrainNN, ValidateNN, TestNN
import logging
import gensim
import torch
import numpy as np
import string

DEVICE="CUDA"
NB_EPOCHS=20
STEP_SIZE=15
GAMMA=0.5


torch.manual_seed(0)
np.random.seed(0)

train_valid_test_set = CsvDataset(
    'input/reviews_amazon_com.csv',
    f_text=lambda review: review['title'] + ' ' + review['body'],
    f_label=lambda review: int(review['stars'])
)
# Preprocess
#   Remove punctuation
train_valid_test_set.processed = [
    doc.translate({ord(p): None for p in string.punctuation})
    for doc in train_valid_test_set.text
]
#   Tokenize by splitting on whitespaces and lowering
train_valid_test_set.processed = [
   doc.lower().split() for doc in train_valid_test_set.processed
]
# Shuffle, then split: train 90%, valid 5%, test 5%.
train_valid_test_set.shuffle()
train_valid_set, test_set = train_valid_test_set.split(0.95)
train_set, valid_set = train_valid_set.split(
    1 - len(test_set) / len(train_valid_set)
)

# Gensim model/embedding using train_set text
embedding = gensim.models.Word2Vec.load('wiki-english.model')
embedding.train(train_set.processed, total_examples=len(train_set), epochs=5)
#   Add <unk> token
embedding.wv.add('<unk>', np.full(embedding.wv.vector_size, 0))

# Find longest document in training set.
longest_document_length = np.max([len(tokens) for tokens in train_set.processed])

# Transform documents into lists of indices to the embedding.
def Word2Int(document_set, embedding, pad_to_length):
    document_set.vec = []
    for doc in document_set.processed:
        document_set.vec.append(torch.tensor(
            [
                embedding.wv.vocab[word].index
                if word in embedding.wv.vocab else -1
                for word in doc
            ] + [-1] * (pad_to_length - len(doc))
        ))

Word2Int(train_set, embedding, longest_document_length)
Word2Int(valid_set, embedding, longest_document_length)
Word2Int(test_set, embedding, longest_document_length)



# LSTM
lstm = SentimentLSTM(embedding.wv, num_classes=5, hidden_size=256, num_layers=2)
lstm_opt = torch.optim.Adam(lstm.parameters(), lr=0.05)
lstm_sch = optim.lr_scheduler.StepLR(lstm_opt, STEP_SIZE, SCH_GAMMA)

# Train
for epoch in range(1, NB_EPOCHS + 1):
    # Display training output
    with grid.output_to(grid.rows-1, 0):
        TrainNN(lstm, "LSTM", DEVICE, train_loader, lstm_opt, epoch)
        validation_results[epoch-1, 0:2] = validate(lstm, "LSTM", DEVICE, valid_loader)
        validation_results[epoch-1, 2] = lstm_opt.param_groups[0]['lr']
        aim_lstm_sch.step()
        # aim_lstm_sch.step(validation_results[epoch-1, 0])

