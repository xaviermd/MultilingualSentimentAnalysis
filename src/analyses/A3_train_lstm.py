from A0_utils import CsvDataset, SentimentLSTM, TrainNN, ValidateNN, TestNN
from torch.utils.data import DataLoader
import logging
import gensim
import torch
import numpy as np
import string
import joblib
import gdown
import os

DEVICE="cuda"
NB_EPOCHS=30
BATCH_SIZE=16
SCH_STEP_SIZE=15
SCH_GAMMA=0.5
LOG_INTERVAL=100


if not os.path.exists("/tmp/xmd/"):
    os.mkdir("/tmp/xmd/")

if not os.path.exists("/tmp/xmd/input"):
    os.mkdir("/tmp/xmd/input")
if not os.path.exists("/tmp/xmd/input/train_valid_test/"):
    os.mkdir("/tmp/xmd/input/train_valid_test/")

filenames_urls = {
  "/tmp/xmd/input/train_valid_test/train_amazon_com.csv": "https://drive.google.com/uc?id=11U_GLfigjC8B4dfXYFU0IqaVHtUlcBl0",
  "/tmp/xmd/input/train_valid_test/valid_amazon_com.csv": "https://drive.google.com/uc?id=1vh94sOv7inWi88jn8Bqc_7MrPej8DyvX",
  "/tmp/xmd/input/train_valid_test/test_amazon_com.csv": "https://drive.google.com/uc?id=1Wa22vKeMORmMTJ8Ci1xY-yYyLUlT137n",
#   "/tmp/xmd/input/train_valid_test/train_amazon_fr.csv": "https://drive.google.com/uc?id=1pFyPkizNxCXXNG03B3jhsxSfAJo3iemV",
#   "/tmp/xmd/input/train_valid_test/valid_amazon_fr.csv": "https://drive.google.com/uc?id=1pAI4S0Hdxysm3zq0rOllUF0j6woAfG_s",
#   "/tmp/xmd/input/train_valid_test/test_amazon_fr.csv": "https://drive.google.com/uc?id=1ecfNsTocUBsHlrauQ1dnwn6Mjt0A796t",
#   "/tmp/xmd/input/train_valid_test/train_amazon_es.csv": "https://drive.google.com/uc?id=16WyzjSpDnCdUHUcQjuIZwGYlsbg1Kizt",
#   "/tmp/xmd/input/train_valid_test/valid_amazon_es.csv": "https://drive.google.com/uc?id=1BXE4ACEQJ9Gpu2cquxrNNjtRJL1S8e2h",
#   "/tmp/xmd/input/train_valid_test/test_amazon_es.csv": "https://drive.google.com/uc?id=1qJJE7Hyj7zN0pygYvYrgCn5B5ERVLdvo",
#   "/tmp/xmd/input/train_valid_test/train_amazon_cn.csv": "https://drive.google.com/uc?id=1yNiombiBFSyE6Otpf4TjIeb7-isDKhO8",
#   "/tmp/xmd/input/train_valid_test/valid_amazon_cn.csv": "https://drive.google.com/uc?id=139LFaDkjvi3kGSuOMBLxyf2BMPfAs3Rm",
#   "/tmp/xmd/input/train_valid_test/test_amazon_cn.csv": "https://drive.google.com/uc?id=1qfBSmoSgeTS6JidZMtHcSjwbJcDC_xbg"
}

for filename, url in filenames_urls.items():
    if not os.path.exists(filename):
        gdown.download(url, filename, False)
    else:
        print('{} already exists. Skipping.'.format(filename))


if not os.path.exists("/tmp/xmd/input"):
    os.mkdir("/tmp/xmd/input")
if not os.path.exists("/tmp/xmd/input/gensim"):
    os.mkdir("/tmp/xmd/input/gensim")

filenames_urls = {
    "/tmp/xmd/input/gensim/amazon-com.wv": "https://drive.google.com/uc?id=1XJa4V8pTkw4otHkfDzdliqO3l5VL87_a",
#     "/tmp/xmd/input/gensim/text8-english.wv": "https://drive.google.com/uc?id=16WV0na21Uin_fXaZB3LK4uDq4xfRZyrP",
#     "/tmp/xmd/input/gensim/text8-amazon-com.wv": "https://drive.google.com/uc?id=18wu9-yqz3ZV1sz1FwIWfySkWyrXcBEqK"
}

for filename, url in filenames_urls.items():
    if not os.path.exists(filename):
        gdown.download(url, filename, False)
    else:
        print('{} already exists. Skipping.'.format(filename))


# Train LSTM
torch.manual_seed(0)
np.random.seed(0)

train_set = CsvDataset('/tmp/xmd/input/train_valid_test/train_amazon_com.csv')
valid_set = CsvDataset('/tmp/xmd/input/train_valid_test/valid_amazon_com.csv')
test_set = CsvDataset('/tmp/xmd/input/train_valid_test/test_amazon_com.csv')

# Gensim model/embedding using train_set text
embedding = gensim.models.KeyedVectors.load('/tmp/xmd/input/gensim/amazon-com.wv', mmap='r')
#   Add <unk> token
embedding.add('<unk>', np.full(embedding.vector_size, 0))

# Find longest document in training set.
longest_document_length = np.max([len(tokens) for tokens in train_set.preprocessed])

# Transform documents into lists of indices to the embedding.
def Word2Ix(document_set, embedding, pad_to_length):
  document_set.vec = []
  for doc in document_set.preprocessed:
    document_set.vec.append(torch.tensor(
      [
        embedding.vocab[word].index
        if word in embedding.vocab else -1
        for word in doc
      ] + [-1] * (pad_to_length - len(doc))
    ))

Word2Ix(train_set, embedding, longest_document_length)
Word2Ix(valid_set, embedding, longest_document_length)
Word2Ix(test_set, embedding, longest_document_length)

# Labels must be 0-4 rather than 1-5
train_set.labels = [label-1 for label in train_set.labels]
valid_set.labels = [label-1 for label in valid_set.labels]
test_set.labels = [label-1 for label in test_set.labels]

# Loaders
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# LSTM
lstm = SentimentLSTM(embedding, num_classes=5, hidden_size=256, num_layers=2).to(DEVICE)
lstm_opt = torch.optim.Adam(lstm.parameters(), lr=0.05)
lstm_sch = torch.optim.lr_scheduler.StepLR(lstm_opt, SCH_STEP_SIZE, SCH_GAMMA)


# data, target = next(iter(train_loader))
# data, target = data.to(DEVICE), target.to(DEVICE)
# #target = target.double().to(DEVICE)
# target = target - 1 #1-5 -> 0.0-4.0
# result = torch.cat([
#     torch.cat([
#         torch.as_tensor(embedding.vectors[word_index]).unsqueeze(1).t()
#         for word_index in sentence
#     ]).unsqueeze(0)
#     for sentence in data
#   ], dim=0).to(torch.float32).to(data.device)
# result, (hidden, cell) = lstm.lstm(result)
# result = lstm.drop(result[:, -1, :])
# result = lstm.classifier(result)
# output = lstm.logsoftmax(result)
# print(output.device)
#
# torch.nn.functional.nll_loss(output, target)
# Train
for epoch in range(1, NB_EPOCHS + 1):
    TrainNN(lstm, "LSTM", DEVICE, train_loader, lstm_opt, epoch, LOG_INTERVAL)
    validation_results[epoch-1, 0:2] = validate(lstm, "LSTM", DEVICE, valid_loader)
    validation_results[epoch-1, 2] = lstm_opt.param_groups[0]['lr']
    aim_lstm_sch.step()
