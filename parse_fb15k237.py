import json
import sys
import urllib
import requests
import csv
import multiprocessing
from multiprocessing import Pool
import numpy as np
from itertools import repeat
import itertools



QUERY_ENTITY = """
    SELECT ?s WHERE {{ ?s wdt:P646 \"{0!s}\" }}
"""

QUERY_RELATION = """
"""

def f(x, query):
    print(multiprocessing.current_process())
    results_yes, results_no = [], []

    count = 1
    for uri in x:
        res = query_kg(query.format(uri))
        if len(res["results"]["bindings"]) > 0:
            wd_uri = res["results"]["bindings"][0]["s"]["value"]
            print(f"{uri} : {wd_uri}, {count}/{len(x)}")
            results_yes.append(wd_uri)
        else:
            results_no.append(uri)
            print(f"No mapping for {uri}")
        count += 1

    return (results_yes, results_no)

def parse_fb15k(path, num_workers):
    s, p = [], []
    wd = []
    nodata = []
    names = ["train.txt", "valid.txt", "test.txt"]
    for n in names:
        with open(path+"/"+n, "r", newline='') as f1:
            dataset = csv.reader(f1, delimiter="\t")

            for line in dataset:
                s.extend([line[0], line[2]])
                if "." in line[1]:
                    p.extend(line[1].split("."))
                else:
                    p.append(line[1])

    unique_entities = list(set(s))
    unique_p = list(set(p))

    print(len(unique_entities))
    print(len(unique_p))


    chunks = np.array_split(unique_entities, num_workers)
    with Pool(processes=num_workers) as pool:
        results = pool.starmap(f, zip(chunks, repeat(QUERY_ENTITY)))
    final_res = list(set(list(itertools.chain(*[i[0] for i in results]))))
    final_neg = list(set(list(itertools.chain(*[i[1] for i in results]))))

    print(f"Found {len(final_res)} / {len(unique_entities)} mappings")
    # for fb_uri in unique_entities:
    #     res = query_kg(QUERY_ENTITY.format(fb_uri))
    #     if len(res["results"]["bindings"]) > 0:
    #         wd_uri = res["results"]["bindings"][0]["s"]["value"]
    #         print(f"{fb_uri} : {wd_uri}")
    #         count += 1
    #         wd.append(wd_uri)
    #     else:
    #         nodata.append(fb_uri)
    #         print(f"No mapping for {fb_uri}")
    #
    #

    with open("fb15k_237_wikidata_entities.txt", "w") as out, open("fb15k_todo.txt", "w") as out2:
        out.write("\n".join(final_res))
        out2.write("\n".join(final_neg))


def query_kg(query):
    address = "http://eis-warpcore-01:5820/wikidata/query"
    params = urllib.parse.urlencode({"query": query, "format": "json"})
    r = requests.get(address, params=params, headers={'Accept': 'application/sparql-results+json'},
                     auth=('admin', 'admin'))
    try:
        results = json.loads(r.text)
        return results
    except Exception as e:
        raise Exception("Smth is wrong with the endpoint", str(e), " , ", r.status_code)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Path to the files with dataset is required")
    parse_fb15k(sys.argv[1], int(sys.argv[2]))