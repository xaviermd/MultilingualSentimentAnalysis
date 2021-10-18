from A0_utils import CsvDataset, Preprocess_EN_FR_ES
import numpy as np
import torch
import joblib
import pynlpir
import os

import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO
)

os.makedirs('input/translated_train_valid_test/', exist_ok=True)

branches = ["amazon.fr", "amazon.es", "amazon.cn"]
pynlpir.open()
for branch in branches:
    dataset_names = ["train", "valid", "test"]
    for dataset in dataset_names:
        torch.manual_seed(0)
        np.random.seed(0)
        csv_dataset = CsvDataset(
            'input/translated/{}_{}_translated.csv'\
              .format(dataset, branch.replace('.', '_')),
            f_text=lambda review: (review["text"], review["translation"]),
            f_label=lambda review: int(review["labels"])
        )
        csv_dataset.original = [original for original, _ in csv_dataset.text]
        csv_dataset.text = [text for _, text in csv_dataset.text]

        # Preprocess
        csv_dataset.preprocessed = [
            Preprocess_EN_FR_ES(doc) for doc in csv_dataset.text
        ]
        # Shuffle
        csv_dataset.shuffle()

        # Save train_set, valid_set, and test_set
        joblib.dump(
            csv_dataset,
            'input/train_valid_test/{}_{}.joblib'.format(
                dataset,
                branch.replace('.', '_')
            )
        )
        csv_dataset.save(
            'input/train_valid_test/{}_{}.csv'.format(
                dataset,
                branch.replace('.', '_')
            ),
            ["labels", "original", "text", "preprocessed"]
        )
