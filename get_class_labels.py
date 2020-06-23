import requests
import urllib
import json
from tqdm import tqdm
from pathlib import Path
from functools import partial
import multiprocessing as mp
import random

QUERY_FULL = """
    SELECT ?uri WHERE {{ wd:{0!s} wdt:P31/wdt:P279* ?uri }}
"""

QUERY_TYPE_ONLY = """
    SELECT ?uri WHERE {{ wd:{0!s} wdt:P31+ ?uri }}
"""

CHECK_P279 = """
    ASK WHERE {{ wd:{0!s} wdt:P279 ?x }}
"""

QUERY_FOR_INSTANCE = """
    SELECT DISTINCT ?uri WHERE {{
      {{ wd:{0!s} wdt:P31 ?uri . }}  # hop1
      UNION {{ wd:{0!s} wdt:P31 / wdt:P279 ?uri }} # hop2
      UNION {{ wd:{0!s} wdt:P31 / wdt:P279 / wdt:P279 ?uri }} # hop3
    }}
"""

QUERY_FOR_CLASS = """
    SELECT DISTINCT ?uri WHERE {{
      {{ wd:{0!s} wdt:P279 ?uri . }} # hop1
      UNION {{ wd:{0!s} wdt:P279 / wdt:P279 ?uri }} # hop2
      UNION {{ wd:{0!s} wdt:P279 / wdt:P279 / wdt:P279 ?uri }} # hop3
    }}
"""

ENDPOINT = "http://eis-warpcore-01:5820/wikidata/query"

def executeQuery(query):
    params = urllib.parse.urlencode({"query": query, "format": "json"})
    r = requests.get(ENDPOINT,
                     params=params,
                     headers={'Accept': 'application/sparql-results+json'},
                     auth=('admin','admin'))
    try:
        results = json.loads(r.text)
        return results
    except Exception:
        raise Exception("Smth is wrong with the endpoint, query was ", query)


def get_entity_label(uri, qtype):
    query = qtype.format(uri)
    res = executeQuery(query)
    if len(res["results"]["bindings"]) == 0:
        return {uri: []}
    else:
        types = sorted([t["uri"]["value"].replace("http://www.wikidata.org/entity/", "")
                 for t in res["results"]["bindings"]
                 ], key=len)
        return {uri: types}


def process_dataset_entities(ds, subtype, qtype):
    DIRNAME = Path(f'data/clean/{ds}/{subtype}')

    with open(DIRNAME / 'train.txt', 'r') as f:
        raw_trn = []
        for line in f.readlines():
            raw_trn.append(line.strip("\n").split(","))

    with open(DIRNAME / 'test.txt', 'r') as f:
        raw_tst = []
        for line in f.readlines():
            raw_tst.append(line.strip("\n").split(","))

    with open(DIRNAME / 'valid.txt', 'r') as f:
        raw_val = []
        for line in f.readlines():
            raw_val.append(line.strip("\n").split(","))

    entities = [elem for statement in raw_trn+raw_val+raw_tst for elem in statement if elem[0] == "Q"]
    entities = sorted(list(set(entities)))

    qtemplate = QUERY_TYPE_ONLY if qtype == "type" else QUERY_FULL
    qfunction = partial(get_entity_label, qtype=qtemplate)

    with mp.Pool() as pool:
        all_batches = tqdm(pool.map(qfunction, entities))
        class_labels = {k: v for batch in all_batches for (k, v) in batch.items()}

    json.dump(class_labels,
              open(DIRNAME / f"class_labels_{qtype}.json", "w"),
              indent=4)

    print("Done")


def clean_classes(ds, subtype):
    print(f"Processing {ds} {subtype}")
    DIRNAME = Path(f'data/clean/{ds}/{subtype}')

    with open(DIRNAME / 'train.txt', 'r') as f:
        raw_trn = []
        for line in f.readlines():
            raw_trn.append(line.strip("\n").split(","))

    with open(DIRNAME / 'test.txt', 'r') as f:
        raw_tst = []
        for line in f.readlines():
            raw_tst.append(line.strip("\n").split(","))

    with open(DIRNAME / 'valid.txt', 'r') as f:
        raw_val = []
        for line in f.readlines():
            raw_val.append(line.strip("\n").split(","))

    # classes = set([s[2] for s in raw_trn+raw_val+raw_tst if s[1] == "P31"])
    # classes.update([s[2] for s in raw_trn+raw_val+raw_tst if s[1] == "P279"])
    classes = set([s[n+1] for s in raw_trn+raw_val+raw_tst for n,e in enumerate(s) if e=="P31" or e=="P279"])
    # cl = set([s[2] for s in raw_trn+raw_val+raw_tst if s[1] == "P31" or s[1] == "P279"])
    # assert len(classes) == len(cl)
    # noc_statements = [s for s in raw_trn+raw_val+raw_tst
    #                   if s[0] not in classes and s[1] != "P31" and s[1] != "P279"]
    noc_statements = [s for s in raw_trn + raw_val + raw_tst
                      if s[0] not in classes and ("P31" not in s[1::2]) and ("P279" not in s[1::2])]

    entities = [elem for statement in noc_statements for elem in statement if elem[0] == "Q"]
    entities = sorted(list(set(entities)))
    rels = [elem for statement in noc_statements for elem in statement if elem[0] == "P"]
    rels = sorted(list(set(rels)))

    with open(DIRNAME / "nc_edges.txt","w") as e_out:
        e_out.writelines(((",").join(l))+'\n' for l in noc_statements)
    with open(DIRNAME / "nc_entities.txt","w") as ent_out:
        ent_out.writelines(l + '\n' for l in entities)
    with open(DIRNAME / "nc_rels.txt","w") as rel_out:
        rel_out.writelines(l + '\n' for l in rels)


def send_p279(e):
    query = CHECK_P279.format(e)
    res = executeQuery(query)
    if res["boolean"] == True:
        return 1
    else:
        return 0

def extract_3hop_hierarchy(e):
    if send_p279(e) == 1:
        # e is a class
        res = get_entity_label(e, QUERY_FOR_CLASS)
        return res
    else:
        # e is an instance
        res = get_entity_label(e, QUERY_FOR_INSTANCE)
        return res

def check_num_classes(ds, subtype):
    print(f"Processing {ds} {subtype}")
    DIRNAME = Path(f'data/clean/{ds}/{subtype}')

    with open(DIRNAME / "nc_entities.txt", "r") as e_in:
        entities = [l.strip("\n") for l in e_in.readlines()]

    count = 0
    with mp.Pool() as pool:
        all_batches = tqdm(pool.imap(send_p279, entities))
        count = sum(all_batches)

    print(f"Dataset has {count} entities with P279 out of total {len(entities)}")


def extract_full_wd50k():
    DIRNAME = Path(f'data/clean/wd15k/statements')

    with open(DIRNAME / "nc_entities.txt", "r") as e_in:
        entities = [l.strip("\n") for l in e_in.readlines()]

    with mp.Pool() as pool:
        all_batches = tqdm(pool.imap(extract_3hop_hierarchy, entities))
        class_labels = {k: v for batch in all_batches for (k, v) in batch.items()}

    json.dump(class_labels,
              open(DIRNAME / f"nc_wf50k_full_class_labels.json", "w"),
              indent=4)

    print("Done")

def extract_specific(ds):
    DIRNAME = Path(f'data/clean/{ds}/statements')

    with open(DIRNAME / "nc_entities.txt", "r") as e_in:
        entities = [l.strip("\n") for l in e_in.readlines()]

    with mp.Pool() as pool:
        all_batches = tqdm(pool.imap(extract_3hop_hierarchy, entities))
        class_labels = {k: v for batch in all_batches for (k, v) in batch.items()}

    json.dump(class_labels,
              open(DIRNAME / f"nc_wf50k_full_class_labels.json", "w"),
              indent=4)

    print("Done")

def obtain_ds_labels(ds):
    # reads a full labels file and creates the same for triples
    full_dump = json.load(open(f"data/clean/{ds}/statements/nc_{ds.replace('15k','50k')}_class_labels.json", "r"))

    for subtype in ["triples"]:
        print(f"Processing {ds} {subtype}")
        DIRNAME = Path(f'data/clean/{ds}/{subtype}')

        with open(DIRNAME / "nc_entities.txt","r") as e_in:
            nc_entities = [e.strip("\n") for e in e_in.readlines()]

        count = 0
        extracted = 0
        ds_entity_labels = {}
        for e in nc_entities:
            if e in full_dump:
                ds_entity_labels[e] = full_dump[e]
            else:
                print(f"{e} labels are not there")
                count += 1
                try:
                    classes = extract_3hop_hierarchy(e)
                    ds_entity_labels[e] = classes[e]
                    print("Extracted")
                    extracted += 1
                except:
                    continue

        print(f"Overall not there: {count}, extracted: {extracted}")
        json.dump(ds_entity_labels,
                  open(DIRNAME / f"nc_class_labels.json", "w"),
                  indent=4)


def create_splits(ds):
    DIRNAME = Path(f'data/clean/{ds}')

    labels_name = ds.replace('15', '50')
    with open(DIRNAME / "statements" / "nc_edges.txt", "r") as sgraph_in,\
        open(DIRNAME / "statements" / f"nc_{labels_name}_class_labels.json", "r") as slabs_in:
        sgraph = [l.strip("\n").split(",") for l in sgraph_in.readlines()]
        slabs = json.load(slabs_in)

    with open(DIRNAME / "triples" / "nc_edges.txt", "r") as tgraph_in,\
        open(DIRNAME / "triples" / f"nc_class_labels.json", "r") as tlabs_in:
        tgraph = [l.strip("\n").split(",") for l in tgraph_in.readlines()]
        tlabs = json.load(tlabs_in)

    statement_all_entities = list(set([e for s in sgraph for e in s[0::2]]))
    statement_so = list(set([e for s in sgraph for e in [s[0], s[2]]]))
    qual_only_entities = list(set(statement_all_entities).difference(set(statement_so)))
    triple_so = list(set([e for t in tgraph for e in [t[0], t[2]]]))

    random.seed(42)
    train_vol, val_vol, test_vol = 0.8, 0.1, 0.1
    # idea: we want to predict the same S and O types in statements/triples
    # split the triple based-dump first
    # create statement splits based on triple splits
    random.shuffle(triple_so)
    triple_train_nodes = triple_so[: int(len(triple_so)*train_vol)]
    triple_val_nodes = triple_so[int(len(triple_so)*train_vol) : int(len(triple_so)*(train_vol+val_vol))]
    triple_test_nodes = triple_so[int(len(triple_so)*(train_vol+val_vol)): ]

    stat_so_train_nodes = [e for e in triple_train_nodes if e in statement_so]
    stat_so_val_nodes = [e for e in triple_val_nodes if e in statement_so]
    stat_so_test_nodes = [e for e in triple_test_nodes if e in statement_so]
    stat_so_rest = list(set(statement_all_entities).difference(set(triple_so)))

    print('next')
    # create another split for statment-only for a general node classification
    random.shuffle(statement_all_entities)
    stat_full_train_nodes = statement_all_entities[: int(len(statement_all_entities)*train_vol)]
    stat_full_val_nodes = statement_all_entities[int(len(statement_all_entities)*train_vol): int(len(statement_all_entities)*(train_vol+val_vol))]
    stat_full_test_nodes = statement_all_entities[int(len(statement_all_entities)*(train_vol+val_vol)): ]

    # create the files
    triple_train = {e: tlabs[e] for e in triple_train_nodes}
    triple_val = {e: tlabs[e] for e in triple_val_nodes}
    triple_test = {e: tlabs[e] for e in triple_test_nodes}
    json.dump(triple_train, open(DIRNAME / "triples" / "nc_train_labels.json","w"), indent=True)
    json.dump(triple_val, open(DIRNAME / "triples" / "nc_val_labels.json", "w"), indent=True)
    json.dump(triple_test, open(DIRNAME / "triples" / "nc_test_labels.json", "w"), indent=True)

    stat_so_train = {e: slabs[e] for e in stat_so_train_nodes}
    stat_so_val = {e: slabs[e] for e in stat_so_val_nodes}
    stat_so_test = {e: slabs[e] for e in stat_so_test_nodes}
    stat_so_rest = {e: slabs[e] for e in stat_so_rest}

    json.dump(stat_so_train, open(DIRNAME / "statements" / "nc_train_so_labels.json","w"), indent=True)
    json.dump(stat_so_val, open(DIRNAME / "statements" / "nc_val_so_labels.json", "w"), indent=True)
    json.dump(stat_so_test, open(DIRNAME / "statements" / "nc_test_so_labels.json", "w"), indent=True)
    json.dump(stat_so_rest, open(DIRNAME / "statements" / "nc_rest_so_labels.json", "w"), indent=True)

    stat_full_train = {e: slabs[e] for e in stat_full_train_nodes}
    stat_full_val = {e: slabs[e] for e in stat_full_val_nodes}
    stat_full_test = {e: slabs[e] for e in stat_full_test_nodes}

    json.dump(stat_full_train, open(DIRNAME / "statements" / "nc_train_full_labels.json", "w"), indent=True)
    json.dump(stat_full_val, open(DIRNAME / "statements" / "nc_val_full_labels.json", "w"), indent=True)
    json.dump(stat_full_test, open(DIRNAME / "statements" / "nc_test_full_labels.json", "w"), indent=True)

    print("Done")


if __name__ == "__main__":
    # process_dataset_entities("wd15k", "statements", "full")
    # process_dataset_entities("wd15k_33", "statements", "full")
    # process_dataset_entities("wd15k_66", "statements", "full")
    # process_dataset_entities("wd15k_qonly", "statements", "full")
    # process_dataset_entities("wd15k", "triples", "full")
    # process_dataset_entities("wd15k_33", "triples", "full")
    # process_dataset_entities("wd15k_66", "triples", "full")
    # process_dataset_entities("wd15k_qonly", "triples", "full")
    # clean_classes("wd15k", "statements")
    # clean_classes("wd15k_33", "statements")
    # clean_classes("wd15k_66", "statements")
    # clean_classes("wd15k_qonly", "statements")
    # clean_classes("wd15k", "triples")
    # clean_classes("wd15k_33", "triples")
    # clean_classes("wd15k_66", "triples")
    # clean_classes("wd15k_qonly", "triples")
    #check_num_classes("wd15k", "statements") # Dataset has 6616 entities with P279 out of total 46164
    #extract_full_wd50k()
    #extract_specific("wd15k_qonly")
    #obtain_ds_labels("wd15k_qonly")
    create_splits("wd15k_qonly")
    print("DONE")