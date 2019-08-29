from tqdm import tqdm_notebook as tqdm
from pprint import pprint
import pickle
import pandas as pd
from pathlib import Path
from collections import namedtuple
Quint = namedtuple('Quint', 's p o qp qe')
RAW_DATA_DIR = Path('./data/raw_data/wd15k')
PARSED_DATA_DIR = Path('./data/parsed_data')
PRETRAINING_DATA_DIR = Path('./data/pre_training_data')


def generate_data(sid_key, sid_value):
    # Get all the rights
    result, qualifiers = [], []
    s, p, o = None, None, None
    for triple in sid_value:
        if triple[0] == sid_key and '/qualifier' in triple[1]:
            qualifiers.append(triple)
        elif triple[0] == sid_key and '/statement' in triple[1]:
            o = triple[2].split('/')[-1].replace('>', '')
        elif triple[-1] == sid_key:
            s, p = triple[0].split('/')[-1].replace('>', ''), triple[1].split('/')[-1].replace('>', '')

    try:
        assert s
        assert p
    except AssertionError:
        raise IOError
    try:
        assert o
    except AssertionError:
        return []
    #         print(sid_key)
    #         for x in sid_value:
    #             print(x)
    #         raise IOError

    if len(qualifiers) > 0:
        for qualifier in qualifiers:
            qp, qe = qualifier[1].split('/')[-1].replace('>', ''), qualifier[2].split('/')[-1].replace('>', '')
            q = Quint(s=s, p=p, o=o, qp=qp, qe=qe)
            result.append(q)
    else:
        q = Quint(s=s, p=p, o=o, qp=None, qe=None)
        result.append(q)

    return result

def generate_full_quints_statement(sid_key, sid_value):
    # Get all the rights
    result, qualifiers = [], []
    s, p, o = None, None, None
    for triple in sid_value:
        if triple[0] == sid_key and '/qualifier' in triple[1]:
            qualifiers.append(triple)
        elif triple[0] == sid_key and '/statement' in triple[1]:
            o = triple[2].split('/')[-1].replace('>', '')
        elif triple[-1] == sid_key:
            s, p = triple[0].split('/')[-1].replace('>', ''), triple[1].split('/')[-1].replace('>', '')

    try:
        assert s
        assert p
    except AssertionError:
        raise IOError
    try:
        assert o
    except AssertionError:
        return []

    statement = {}
    statement['s'], statement['p'], statement['o'] = s, p, o
    statement['qualifiers'] = []
    if len(qualifiers) > 0:
        for qualifier in qualifiers:
            qp, qe = qualifier[1].split('/')[-1].replace('>', ''), qualifier[2].split('/')[-1].replace('>', '')
            statement['qualifiers'].append((qp, qe))

    return statement

def generate_stdreif(sid_key, sid_value):
    result, qualifiers = {}, []
    s, p, o = None, None, None
    for triple in sid_value:
        if triple[0] == sid_key and '/qualifier' in triple[1]:
            qualifiers.append(triple)
        elif triple[0] == sid_key and '/statement' in triple[1]:
            o = triple[2].split('/')[-1].replace('>', '')
        elif triple[-1] == sid_key:
            s, p = triple[0].split('/')[-1].replace('>', ''), triple[1].split('/')[-1].replace('>', '')

    try:
        assert s
        assert p
    except AssertionError:
        raise IOError
    try:
        assert o
    except AssertionError:
        return []

    result["sid"] = sid_key
    result["rdf:subject"], result["rdf:predicate"], result["rdf:object"] = s, p, o
    if len(qualifiers) > 0:
        for qualifier in qualifiers:
            qp, qe = qualifier[1].split('/')[-1].replace('>', ''), qualifier[2].split('/')[-1].replace('>', '')
            result[qp] = qe

    return result



if __name__ == "__main__":

    raw_data = []
    with open(RAW_DATA_DIR / 'fb15k_wd_uri_only.nt', 'r') as f:
        for line in f.readlines():
            raw_data.append(line)

    sids = {}
    for triples in raw_data:
        triples = triples.replace('\n', '').replace(' .', '').split()
        for thing in triples:
            if 'statement/Q' in thing:
                relevant_triples = sids.get(thing, [])
                relevant_triples.append(triples)
                sids[thing] = relevant_triples

    parsed_data = []
    skipped = 0

    ## generate flat quints
    for sid_key, sid_value in tqdm(sids.items()):
        res = generate_data(sid_key, sid_value)
        if res == []:
            skipped += 1
            continue
        parsed_data += res


    with open(PARSED_DATA_DIR / 'parsed_raw_data.pkl', 'wb+') as f:
        pickle.dump(parsed_data, f)

    df = pd.DataFrame(parsed_data)
    df.to_csv(PARSED_DATA_DIR / 'parsed_raw_data.csv', index=False)


    template1 = "<< {0!s} {1!s} {2!s} >> {3!s} {4!s} . \n"
    template2 = "<< {0!s} {1!s} {2!s} >> . \n"
    with open(PARSED_DATA_DIR / "parsed_raw_data.rs", "w") as f:
        for row in parsed_data:
            if row[3]!= None:
                f.write(template1.format(row[0], row[1], row[2], row[3], row[4]))
            else:
                f.write(template2.format(row[0], row[1], row[2]))

    ## generate good quints
    good_quints = []
    for sid_key, sid_value in tqdm(sids.items()):
        res = generate_full_quints_statement(sid_key, sid_value)
        if res == []:
            continue
        good_quints.append(res)
    templ = """ << {0!s} {1!s} {2!s} >> {3!s} . \n"""

    """
    <<  s p o  >> q1 v1;
                q2 v2;
                qn, vn .
    """

    with open(PARSED_DATA_DIR / "parsed_raw_full_quints_statements.rs", "w") as f:
        for row in good_quints:
            qual_string = " ;\n\t".join(f"{qp} {qe}" for qp,qe in row["qualifiers"])
            f.write(templ.format(row['s'], row['p'], row['o'], qual_string))

    with open(PARSED_DATA_DIR / "parsed_raw_full_quints_statements.pkl", "wb+") as f:
        pickle.dump(good_quints, f)


    ## generate stdreif
    stdreif = []
    for sid_key, sid_value in tqdm(sids.items()):
        res = generate_stdreif(sid_key, sid_value)
        if res == []:
            continue
        stdreif.append(res)

    with open(PARSED_DATA_DIR / "parsed_raw_data_stdreif.pkl", "wb+") as f:
        pickle.dump(stdreif, f)

    with open(PARSED_DATA_DIR / "parsed_raw_data_stdreif.nt", "w") as f:
        for row in stdreif:
            key = row["sid"].strip("<>")
            for p in list(row.keys()):
                if p != "sid":
                    f.write(f"{key} {p} {row[p]} . \n")