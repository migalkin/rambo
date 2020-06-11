import requests
import urllib
import json
from tqdm import tqdm
from pathlib import Path
from functools import partial
import multiprocessing as mp

QUERY_FULL = """
    SELECT ?uri WHERE {{ wd:{0!s} wdt:P31/wdt:P279* ?uri }}
"""

QUERY_TYPE_ONLY = """
    SELECT ?uri WHERE {{ wd:{0!s} wdt:P31+ ?uri }}
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
        all_batches = tqdm(pool.map(qfunction, entities[:5]))
        class_labels = {k: v for batch in all_batches for (k, v) in batch.items()}

    json.dump(class_labels,
              open(DIRNAME / f"class_labels_{qtype}.json", "w"),
              indent=4)

    print("Done")


if __name__ == "__main__":
    process_dataset_entities("wd15k", "statements", "full")
    process_dataset_entities("wd15k_33", "statements", "full")
    process_dataset_entities("wd15k_66", "statements", "full")
    process_dataset_entities("wd15k_qonly", "statements", "full")
    process_dataset_entities("wd15k", "triples", "full")
    process_dataset_entities("wd15k_33", "triples", "full")
    process_dataset_entities("wd15k_66", "triples", "full")
    process_dataset_entities("wd15k_qonly", "triples", "full")
    print("DONE")