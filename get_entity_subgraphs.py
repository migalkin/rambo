from multiprocessing import Pool, Process
from itertools import repeat
import multiprocessing
import os
import sys
import json
import numpy as np
import urllib
import requests
import itertools
import codecs

QUERY_ENT_METADATA_DIRECT = """
    CONSTRUCT {{ {0!s} ?p ?o }} where {{ {0!s} ?p ?o FILTER (regex(str(?p), "direct/")) }}
"""

QUERY_ENT_STATEMENTS = """
    CONSTRUCT {{  {0!s} ?p ?o . ?o ?p1 ?o1 }} WHERE {{
      {0!s} ?p ?o .
      ?o ?p1 ?o1 FILTER (regex(str(?p1), "statement/") || regex(str(?p1), "qualifier/P") ) 
    }}
"""

QUERY_ENT_STATEMENTS_ONLY_URIS = """
    CONSTRUCT {{  {0!s} ?p ?o . ?o ?p1 ?o1 }} WHERE {{
      {0!s} ?p ?o .
      ?o ?p1 ?o1 FILTER (regex(str(?p1), "statement/") || regex(str(?p1), "qualifier/P") ) .
      ?o1 a wikibase:Item .
    }}
"""

def query_kg(query):
    address = "http://eis-warpcore-01:5820/wikidata/query"
    params = urllib.parse.urlencode({"query": query, "format": "json"})
    r = requests.get(address, params=params, headers={'Accept':'application/n-triples'}, auth=('admin', 'admin'))
    try:
        results = r.text
        return results
    except Exception as e:
        raise Exception("Smth is wrong with the endpoint", str(e), " , ", r.status_code)

def f(x, query):
    print(multiprocessing.current_process())
    results = ""

    count = 0
    for uri in x:
        clean_uri = uri.strip("\n")
        try:
            res = query_kg(query.format(clean_uri))
            num_triples = res.count("\n")
            count += 1
            print(f"{clean_uri} : {num_triples} triples found, {count} / {len(x)}")
            results += res
        except Exception as e:
            print("Problems with URI ", clean_uri, " : ", str(e))

    return results

def parallel_sample(path, num_workers, q):

    if q == "statements":
        query = QUERY_ENT_STATEMENTS
    elif q == "direct":
        query = QUERY_ENT_METADATA_DIRECT
    else:
        query = QUERY_ENT_METADATA_DIRECT

    print(f"Getting {q}")


    with open(path, "r") as f1:
        dataset = [i.strip("\n") for i in f1.readlines()]
        chunks = np.array_split(dataset, num_workers)
        with Pool(processes=num_workers) as pool:
            ls = pool.starmap(f, zip(chunks, repeat(query)))
        #final_res = list(set(list(itertools.chain(*ls))))
        with open(f"fb15k_wd_{q}.nt", "w") as output:
            output.write("".join(ls))
        print("Done")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Path to the files with dataset is required")
    parallel_sample(sys.argv[1], 28, "statements")
