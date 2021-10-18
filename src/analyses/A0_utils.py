from torch.utils.data import Dataset
from datetime import datetime
import numpy as np
import smart_open
import pynlpir
import string
import torch
import gdown
import json
import copy
import csv
import os


def DownloadAmazonDataset():
    # Create input/ directory if it does not already exist
    try:
        os.mkdir('input')
    except FileExistsError:
        pass

    # Download CSV files
    filenames_urls = {
        "input/reviews_amazon_com.csv":
            "https://drive.google.com/uc?id=1-qBVieunBtmcnXOJ_8C0CBGB8du8uu3W",
        "input/reviews_amazon_fr.csv":
            "https://drive.google.com/uc?id=1u5pXbF8aasgiGlNu11CBoBIBcIg-R6Mi",
        "input/reviews_amazon_es.csv":
            "https://drive.google.com/uc?id=1oWvAtgsFN2XILfweScn1verzKuCDOF18",
        "input/reviews_amazon_cn.csv":
            "https://drive.google.com/uc?id=1PrRjFjBywwhYYFql9QEA1W53lLa4LWK0",
    }
    for filename, url in filenames_urls.items():
        if not os.path.isfile(filename):
            gdown.download(url, filename, 0)
        else:
            print('File {} already exists. Continuing.'.format(filename))

    print("done")


class CsvDataset(Dataset):

    def __init__(self, csvfilename=None, f_text=None, f_label=None):

        self.vec = None

        if csvfilename is not None:
            # Read CSV file
            with open(csvfilename, 'r') as csvfile:
                csvreader = csv.DictReader(csvfile, quoting=csv.QUOTE_ALL)
                self.csv = list(csvreader)

            if isinstance(f_text, str):
                text_field = f_text
                f_text = lambda c: c[text_field]
            self.text = [f_text(c) for c in self.csv]

            self.preprocessed = copy.deepcopy(self.text)

            if isinstance(f_label, str):
                label_field = f_label
                f_label = lambda c: c[label_field]
            self.labels = [f_label(c) for c in self.csv]

        else:
            self.csv = []
            self.text = []
            self.preprocessed = []
            self.labels = []

    def shuffle(self):
        # Order the data such that the classes are ordered and proportionally
        # represented (e.g., if for each element from categories 1 and 2 there
        # are two from category 3, then [1, 2, 3, 3, 1, 2, 3, 3, ...])
        # but such that the instances of within the categories are random.
        old_text = self.text
        old_processed = self.preprocessed
        old_vec = self.vec
        old_labels = self.labels
        tmp_dict = {
            lab: ix
            for ix, lab in enumerate(set(self.labels))
        }
        L = np.array([
            tmp_dict.get(lab)
            for lab in self.labels
        ])

        index = [np.where(L == ii)[0] for ii in set(L)]
        permuted = [np.random.permutation(ii.size) for ii in index]

        # CATegory ID's Sorted by category Size (number of elements in the category)
        cat_id_ss = np.argsort([c.size for c in index])

        ix = 0;
        ct = [0]*len(index)
        self.text = [[]]*len(self.text)
        self.preprocessed = [[]]*len(self.preprocessed)
        if self.vec is not None:
            self.vec = [[]]*len(self.vec)
        self.labels = [[]]*len(self.labels)

        for ii in range(len(L)):
            # Current category
            cix = cat_id_ss[ix]
            # Next category
            nix = cat_id_ss[(1 + ix) % len(cat_id_ss)]
            # Finished with a category? Remove it from the list of ID's
            while ct[cix] >= len(permuted[cix]):
                # remove category
                cat_id_ss = np.delete(cat_id_ss, ix)
                # recalculate ix
                ix = ix % len(cat_id_ss)
                # recalculate current and next category
                cix = cat_id_ss[ix]
                nix = cat_id_ss[(1 + ix) % len(cat_id_ss)]

            # Array index (within array index [see below])
            ax = permuted[cix][ct[cix]]
            # Permuted array index
            px = index[cix][ax]

            # Labels = OldLabels[permuted_array_index]
            self.labels[ii] = old_labels[px]
            # Text = OldText[permuted_array_index]
            self.text[ii] = old_text[px]
            self.preprocessed[ii] = old_processed[px]
            if self.vec is not None:
                self.vec[ii] = old_vec[px]
            # Count this as one more instance of this category
            ct[cix] = 1 + ct[cix]
            # How do the category proportions compare? If the category has a higher proportion
            # of elements in the resulting array, move on to the next category.
            if ct[cix] / len(index[cix]) >= ct[nix] / len(index[nix]) :
                ix = (1 + ix) % len(cat_id_ss)


    def split(self, training_validation_split):
        idx = int(np.ceil(len(self) * training_validation_split))

        train = CsvDataset()
        train.csv = self.csv[:idx]
        train.text = self.text[:idx]
        train.preprocessed = self.preprocessed[:idx]
        train.labels = self.labels[:idx]

        validation = CsvDataset()
        validation.csv = self.csv[idx:]
        validation.text = self.text[idx:]
        validation.preprocessed = self.preprocessed[idx:]
        validation.labels = self.labels[idx:]

        if self.vec is not None:
            train.vec = self.vec[:idx]
            validation.vec = self.vec[idx:]

        return train, validation

    def save(self, filename, fields):
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.DictWriter(
                csvfile,
                fieldnames=fields,
                quoting=csv.QUOTE_NONNUMERIC
            )
            csvwriter.writeheader()
            for ix in range(len(self)):
                dct = {}
                for fld in fields:
                    data = self.__dict__[fld][ix]
                    if isinstance(data, list):
                        data = " ".join(data)
                    dct.update({fld: data})
                csvwriter.writerow(dct)


    def __len__(self):
        return len(self.text)


    def __getitem__(self, idx):
        if 0 > idx or idx >= len(self):
            raise IndexError

        if self.vec is not None:
            t = self.vec[idx]
        else:
            t = self.preprocessed[idx]

        if self.labels is None:
            label = idx
        else:
            label = self.labels[idx]

        return (t, label)


    def __setitem__(self, idx, value):
        if not isinstance(value, tuple):
            raise TypeError

        if isinstance(value[0], torch.Tensor):
            self.vec[idx] = value[0];
        else:
            self.preprocessed[idx] = value[0];

        if self.labels is not None:
            self.labels[idx] = value[1]


def Preprocess_EN_FR_ES(doc):
    return [
        word
        for word in doc \
          .translate({ord(p): None for p in string.punctuation}) \
          .lower() \
          .split()
        if word.isalnum()   # remove numbers [DISABLED]
    ]


def Preprocess_CN(doc):
    return \
    [
        token
        for token in pynlpir.segment(
            ''.join([t if t.isalnum() else ' ' for t in doc]),
            pos_tagging=False
        ) if not token.isspace()
    ]


# class PreprocessedWiki(object):
#     def __init__(self, fn):
#         self.fn = fn
#
#     def __iter__(self):
#         with smart_open.open(self.fn, 'rb', encoding="utf-8") as infile:
#             for row in infile:
#                 section = json.loads(row)
#                 for paragraph in section['section_texts']:
#                     yield Preprocess(paragraph)


class SentimentLSTM(torch.nn.Module):
#
    def __init__(self, word_vectors, num_classes, hidden_size, num_layers):
        super(SentimentLSTM, self).__init__()
#         self.word_vectors = word_vectors
        self.embed = torch.nn.Embedding.from_pretrained(
            torch.as_tensor(
                word_vectors,
                dtype=torch.float32
            ),
            freeze=True
        )
        self.lstm = torch.nn.LSTM(
            input_size=word_vectors.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=0.5,
            batch_first=True
        )
        # dropout parameter of LSTM does not add on last layer
        self.drop = torch.nn.Dropout(0.5)
        self.classifier = torch.nn.Linear(hidden_size * 2, num_classes)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.dtype = next(iter(self.lstm._parameters.values())).dtype
#
    def forward(self, sentences_array_of_int):
        # Get 3d matrix of word vector representations
#         result = torch.cat([
#             torch.cat([
#                 torch.as_tensor(self.word_vectors.vectors[word_index]).unsqueeze(1).t()
#                 for word_index in sentence
#             ]).unsqueeze(0)
#             for sentence in sentences_array_of_int
#         ], dim=0).to(self.dtype).to(sentences_array_of_int.device)
        # Forward 3d matrix through model and hope for the best!
        result = self.embed(sentences_array_of_int)
        result, _ = self.lstm(result)
        result = self.drop(result[:, -1, :])
        result = self.classifier(result)
        return self.logsoftmax(result)


class SentimentLSTM_2(torch.nn.Module):
    def __init__(self, word_vectors, num_classes, hidden_size, num_layers):
        super(SentimentLSTM, self).__init__()
        self.embed = torch.nn.Embedding.from_pretrained(
            torch.as_tensor(
                word_vectors,
                dtype=torch.float32
            ),
            freeze=False
        )
        self.lstm = torch.nn.LSTM(
            word_vectors.shape[1],
            hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.5
        )
        self.drop = torch.nn.Dropout(0.5)
        self.classifier = torch.nn.Linear(hidden_size * 2, num_classes)

    def forward(self, text):
        embedded = self.drop(self.embed(text))
        packed_output, (hidden, cell) = self.lstm(embedded)
        hidden = self.drop(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        return torch.nn.functional.log_softmax(self.classifier(hidden), dim=1)


class SentimentCNN(torch.nn.Module):
    def __init__(self,
        num_classes, vocab_size, embedding_size,
        filter_sizes, num_filter, #, l2_reg_lambda
    ):
        super(SentimentCNN, self).__init__()
        self.embed = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=None)
        self.CBRs = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(1, num_filter, (fsize, embedding_size), stride=1),
                torch.nn.BatchNorm2d(embedding_size),
                torch.nn.ReLU()
            ) \
            for fsize in filter_sizes
        ])
        self.drop = torch.nn.Dropout(0.5)
        self.classifier = torch.nn.Linear(len(filter_sizes) * num_filter, num_classes)

    def forward(self, x):
        # Word index to embedded format
        x = self.embed(x)
        # Add channel dimension
        x = torch.unsqueeze(x, 1)
        z = torch.zeros((x.shape[0], 0, 1), device=x.device)
        for cbr in self.CBRs:
            y = cbr(x)
            y = torch.squeeze(y, -1)
            # Remove last dimension (size 1)
            y = torch.squeeze(y, -1)
            y = torch.nn.functional.max_pool2d(y, (1, y.size(2)))
            z = torch.cat((z, y), 1)

        z = self.drop(z)
        z = z.squeeze(-1)
        z = self.classifier(z)
        return torch.nn.functional.log_softmax(z, dim=1)


def TrainNN(model, model_name, device, train_loader, optimizer, epoch, log_interval):
    datetime_start = datetime.now()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            nb_sent_learned = (1 + batch_idx) * len(data)
            nb_sent_remaining = len(train_loader.dataset) - nb_sent_learned
            time_ellapsed = datetime.now() - datetime_start
            print(
                '{{{}}} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                '\tETA: {}'.format(
                    model_name, epoch, batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(),
                    nb_sent_remaining * time_ellapsed / nb_sent_learned
                )
            )


def ValidateNN(model, model_name, device, validation_loader):
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            validation_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    validation_loss /= len(validation_loader.dataset)

    print(
        '\n{{{}}} Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
        .format(
            model_name, validation_loss, correct,
            len(validation_loader.dataset),
            100. * correct / len(validation_loader.dataset)
        )
    )

    return validation_loss, 100. * correct / len(validation_loader.dataset)


def TestNN(model, model_name, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            predictions.append((pred, target))

    test_loss /= len(test_loader.dataset)

    print(
        '\n{{{}}} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
        .format(
            model_name, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)
        )
    )

    return test_loss, 100. * correct / len(test_loader.dataset), predictions



