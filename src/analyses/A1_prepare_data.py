from A0_utils import CsvDataset, Preprocess_EN_FR_ES, Preprocess_CN
import numpy as np
import torch
import joblib
import pynlpir

import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO
)

branches_preprocess = \
{
    'amazon.com': Preprocess_EN_FR_ES,
    'amazon.fr': Preprocess_EN_FR_ES,
    'amazon.es': Preprocess_EN_FR_ES,
    'amazon.cn': Preprocess_CN
}
pynlpir.open()
for branch, preprocess in branches_preprocess.items():
    torch.manual_seed(0)
    np.random.seed(0)

    train_valid_test_set = CsvDataset(
        'input/reviews/reviews_{}.csv'.format(branch.replace('.', '_')),
        f_text=lambda review: review['title'] + ' ' + review['body'],
        f_label=lambda review: int(review['stars'])
    )
    # Preprocess
    train_valid_test_set.preprocessed = [
        preprocess(doc) for doc in train_valid_test_set.preprocessed
    ]
    # Shuffle, then split: train 90%, valid 5%, test 5%.
    train_valid_test_set.shuffle()
    train_valid_set, test_set = train_valid_test_set.split(0.95)
    train_set, valid_set = train_valid_set.split(
        1 - len(test_set) / len(train_valid_set)
    )
    # Save train_set, valid_set, and test_set
    joblib.dump(
        train_set,
        'input/train_valid_test/train_{}.joblib'
            .format(branch.replace('.', '_'))
    )
    train_set.save(
        'input/train_valid_test/train_{}.csv'.format(branch.replace('.', '_')),
        ["labels", "text", "preprocessed"]
    )
    joblib.dump(
        valid_set,
        'input/train_valid_test/valid_{}.joblib'
            .format(branch.replace('.', '_'))
    )
    valid_set.save(
        'input/train_valid_test/valid_{}.csv'.format(branch.replace('.', '_')),
        ["labels", "text", "preprocessed"]
    )
    joblib.dump(
        test_set,
        'input/train_valid_test/test_{}.joblib'
            .format(branch.replace('.', '_'))
    )
    test_set.save(
        'input/train_valid_test/test_{}.csv'.format(branch.replace('.', '_')),
        ["labels", "text", "preprocessed"]
    )
