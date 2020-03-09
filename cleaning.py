from pathlib import Path
import collections
import json
from typing import Dict
from load import _conv_to_our_format_, remove_dups, _get_uniques_


def dedup_wikipeople():
    """
        :return: train/valid/test splits for the wikipeople dataset in its quints form
    """
    DIRNAME = Path('./data/raw_data/wikipeople')
    OUTDIR = Path('./data/clean/wikipeople')

    # Load raw shit
    with open(DIRNAME / 'n-ary_train.json', 'r') as f:
        raw_trn = []
        for line in f.readlines():
            raw_trn.append(json.loads(line))

    with open(DIRNAME / 'n-ary_test.json', 'r') as f:
        raw_tst = []
        for line in f.readlines():
            raw_tst.append(json.loads(line))

    with open(DIRNAME / 'n-ary_valid.json', 'r') as f:
        raw_val = []
        for line in f.readlines():
            raw_val.append(json.loads(line))

    # raw_trn[:-10], raw_tst[:10], raw_val[:10]
    # Conv data to our format
    conv_trn, conv_tst, conv_val = _conv_to_our_format_(raw_trn, filter_literals=True), \
                                   _conv_to_our_format_(raw_tst, filter_literals=True), \
                                   _conv_to_our_format_(raw_val, filter_literals=True)

    conv_trn = list(collections.Counter(tuple(x) for x in conv_trn).keys())
    conv_val = list(collections.Counter(tuple(x) for x in conv_val).keys())
    conv_tst = list(collections.Counter(tuple(x) for x in conv_tst).keys())
    # conv_trn, conv_tst, conv_val = remove_dups(conv_trn), remove_dups(conv_tst), remove_dups(conv_val)


    count = 0
    to_remove = []
    test_spos = set([(i[0], i[1], i[2]) for i in conv_tst])
    for i in conv_trn + conv_val:
        main_triple = (i[0], i[1], i[2])
        if main_triple in test_spos:
            count += 1
            to_remove.append(i)
    print(f"Removing {count} statements from train and val")
    conv_trn = [x for x in conv_trn if x not in to_remove]
    conv_val = [x for x in conv_val if x not in to_remove]

    count = 0
    to_remove = []
    val_spos = set([(i[0], i[1], i[2]) for i in conv_val])
    for i in conv_trn:
        main_triple = (i[0], i[1], i[2])
        if main_triple in val_spos:
            count += 1
            to_remove.append(i)
    print(f"Removing {count} statements from train")
    conv_trn = [x for x in conv_trn if x not in to_remove]

    train_ents = set([item for x in conv_trn for item in x[0::2]])
    train_rels = set([item for x in conv_trn for item in x[1::2]])
    val_ents = set([item for x in conv_val for item in x[0::2]])
    val_rels = set([item for x in conv_val for item in x[1::2]])
    tv_ents = set([item for x in conv_trn + conv_val for item in x[0::2]])
    tv_rels = set([item for x in conv_trn + conv_val for item in x[1::2]])
    test_ents = set([item for x in conv_tst for item in x[0::2]])
    test_rels = set([item for x in conv_tst for item in x[1::2]])

    # clean test first
    ts_unique = test_ents.difference(tv_ents)
    ts_unique_rel = test_rels.difference(tv_rels)
    senseless_triples = []
    for x in conv_tst:
        xe = set(x[0::2])
        xr = set(x[1::2])
        if len(xe.intersection(ts_unique)) > 0:
            senseless_triples.append(x)
            continue
        elif len(xr.intersection(ts_unique_rel)) > 0:
            senseless_triples.append(x)
            continue

    # remove senseless triples from the test
    print(f"Removing {len(senseless_triples)} statements from test")
    conv_tst = [x for x in conv_tst if x not in senseless_triples]

    # # clean valid then
    # ts_unique = val_ents.difference(train_ents)
    # ts_unique_rel = val_rels.difference(train_rels)
    # senseless_triples = []
    # for x in conv_val:
    #     xe = set(x[0::2])
    #     xr = set(x[1::2])
    #     if len(xe.intersection(ts_unique)) > 0:
    #         senseless_triples.append(x)
    #         continue
    #     elif len(xr.intersection(ts_unique_rel)) > 0:
    #         senseless_triples.append(x)
    #         continue
    #
    # # remove senseless triples from the valid
    # print(f"Removing {len(senseless_triples)} statements from val")
    # conv_val = [x for x in conv_val if x not in senseless_triples]

    with open(OUTDIR / "train.txt","w") as train_clean:
        for item in conv_trn:
            train_clean.write(",".join(item)+"\n")

    with open(OUTDIR / "valid.txt","w") as val_clean:
        for item in conv_val:
            val_clean.write(",".join(item)+"\n")

    with open(OUTDIR / "test.txt","w") as test_clean:
        for item in conv_tst:
            test_clean.write(",".join(item)+"\n")

    print("Done")

def clean_jf17k():
    DIRNAME = Path('./data/parsed_data/jf17k')
    OUTDIR = Path('./data/clean/jf17k')

    training_statements = []
    test_statements = []

    with open(DIRNAME / 'train.txt', 'r') as train_file, \
            open(DIRNAME / 'test.txt', 'r') as test_file:

        for line in train_file:
            training_statements.append(line.strip("\n").split(","))

        for line in test_file:
            test_statements.append(line.strip("\n").split(","))

    count = 0
    to_remove = []
    test_spos = set([(i[0], i[1], i[2]) for i in test_statements])
    for i in training_statements:
        main_triple = (i[0], i[1], i[2])
        if main_triple in test_spos:
            count += 1
            to_remove.append(i)

    train = [x for x in training_statements if x not in to_remove]
    print(f"Old len: {len(training_statements)}, new len: {len(train)}")

    train_ents = set([item for x in train for item in x[0::2]])
    train_rels = set([item for x in train for item in x[1::2]])
    test_ents = set([item for x in test_statements for item in x[0::2]])
    test_rels = set([item for x in test_statements for item in x[1::2]])

    # clean test first
    ts_unique = test_ents.difference(train_ents)
    ts_unique_rel = test_rels.difference(train_rels)
    senseless_triples = []
    for x in test_statements:
        xe = set(x[0::2])
        xr = set(x[1::2])
        if len(xe.intersection(ts_unique)) > 0:
            senseless_triples.append(x)
            continue
        elif len(xr.intersection(ts_unique_rel)) > 0:
            senseless_triples.append(x)
            continue

    # remove senseless triples from the test
    print(f"Removing {len(senseless_triples)} statements from test")
    conv_tst = [x for x in test_statements if x not in senseless_triples]

    with open(OUTDIR / "train.txt", "w") as train_clean:
        for datum in train:
            train_clean.write(",".join(datum)+"\n")
    with open(OUTDIR / "test.txt", "w") as test_clean:
        for datum in conv_tst:
            test_clean.write(",".join(datum)+"\n")



if __name__ == "__main__":
    clean_jf17k()