from A0_utils import CsvDataset
import os

if not os.path.exists("input/to_translate/"):
    os.mkdir("input/to_translate");

branches = \
[
    "amazon.com",
    "amazon.fr",
    "amazon.es",
    "amazon.cn"
]
for branch in branches:
    branch = branch.replace('.', '_')
    train_set = CsvDataset(
        "input/train_valid_test/train_{}.csv".format(branch),
        lambda c: c['text'], lambda c: int(c['labels'])
    )
    train_set.text = train_set.text[:5000]
    train_set.preprocessed = train_set.preprocessed[:5000]
    train_set.labels = train_set.labels[:5000]
    train_set.save(
        "input/to_translate/train_{}_to_translate.csv".format(branch),
        ["labels", "text"]
    )

    valid_set = CsvDataset(
        "input/train_valid_test/valid_{}.csv".format(branch),
        lambda c: c['text'], lambda c: int(c['labels'])
    )
    valid_set.text = valid_set.text[:500]
    valid_set.preprocessed = valid_set.preprocessed[:500]
    valid_set.labels = valid_set.labels[:500]
    valid_set.save(
        "input/to_translate/valid_{}_to_translate.csv".format(branch),
        ["labels", "text"]
    )

    test_set = CsvDataset(
        "input/train_valid_test/test_{}.csv".format(branch),
        lambda c: c['text'], lambda c: int(c['labels'])
    )
    test_set.text = test_set.text[:500]
    test_set.preprocessed = test_set.preprocessed[:500]
    test_set.labels = test_set.labels[:500]
    test_set.save(
        "input/to_translate/test_{}_to_translate.csv".format(branch),
        ["labels", "text"]
    )


