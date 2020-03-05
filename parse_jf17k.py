'''

Transform the JF17k dataset into the wikipeople dataset format.
Using the codebase of NaLP (https://github.com/gsp2014/NaLP)

'''

from pathlib import Path

RAW_DATA_DIR = Path('./data/raw_data/jf17k/from_NaLP')
PARSED_DATA_DIR = Path('./data/parsed_data/jf17k')


def parse_and_read(file_name:Path):
    data = []
    with open(file_name, 'r') as f:
        for line in enumerate(f):
            data.append(line[1][:-1].split('\t'))
    return data



# Step 1: read through all relations.
train_data = parse_and_read(RAW_DATA_DIR / 'train.txt')
test_data = parse_and_read(RAW_DATA_DIR / 'test.txt')
all_rels = list(set([value[0] for value in train_data]))
all_rels = list(set(all_rels + list(set([value[1] for value in test_data]))))


def generate_data(data,new_rels, skip_first=False):
    new_data = []
    for node in data:
        if skip_first:
            node = node[1:]
        rels = new_rels[node[0]]
        folds = len(node[1:])

        if folds == 2:
            new_data.append([node[1], rels[0], node[2]])
        if folds == 3:
            new_data.append([node[1], rels[0], node[2], rels[1], node[3]])
        if folds == 4:
            new_data.append([node[1], rels[0], node[2], rels[1], node[3], rels[2], node[4]])
        if folds == 5:
            new_data.append(
                [node[1], rels[0], node[2], rels[1], node[3], rels[2], node[4], rels[3], node[5]])
        if folds == 6:
            new_data.append(
                [node[1], rels[0], node[2], rels[1], node[3], rels[2], node[4], rels[3], node[5],
                 rels[4], node[6]])
        if folds > 6:
            print(f"Error. Folds not defined enough {folds}")
    return new_data


# Step 2: Expand relations
new_rels = {r:[r+'1',r+'2',r+'3',r+'4', r+'5', r+'6'] for r in all_rels}

# Step 3: create new data
new_train_data = generate_data(train_data,new_rels,False)
new_test_data = generate_data(test_data, new_rels,True)



# print number of rels used
used_rels = []
for node in (new_train_data+new_test_data):
    for index, n in enumerate(node):
        if index%2 != 0:
            used_rels.append(n)
print(f"total number of relations now are {len(list(set(used_rels)))}")


with open(PARSED_DATA_DIR / 'train.txt', 'w+') as f:
    for node in new_train_data:
        text = ",".join(node) + '\n'
        f.write(text)

with open(PARSED_DATA_DIR / 'test.txt', 'w+') as f:
    for node in new_test_data:
        text = ",".join(node) + '\n'
        f.write(text)








