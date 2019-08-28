import json
from pathlib import Path
import pickle
from parse_wd15k import Quint

def parse_wikipeople():
    """
    Assuming there is a path ./data/WikiPeople
    :return: train/val/test in RDF*
    """
    templ = """ << {0!s} {1!s} {2!s} >> {3!s} . \n"""
    SOURCE_DIR = Path("./data/WikiPeople")
    try:
        files = list(SOURCE_DIR.glob("*.json"))
        print(files)
        for f in files:
            to_pkl = str(f).split(".json")[0]+"_good_quints.pkl"
            outputname = str(f).split(".json")[0]+"_good_quints.rs"
            to_dump = []
            with open(str(f), "r") as sourcef, open(outputname, "w") as o:
                for line in sourcef:
                    stat_obj = {}
                    stat_obj["qualifiers"] = []
                    statement = json.loads(line)
                    for k in list(statement.keys()):
                        if "_h" in k :
                            stat_obj["subject"] = statement[k]
                            stat_obj["predicate"] = k.split("_")[0]
                        elif '_t' in k :
                            stat_obj["object"] = statement[k]
                        elif k != "N":
                            stat_obj["qualifiers"].append((k, statement[k]))
                    to_dump.append(stat_obj)
                    qual_string = " ; \n\t ".join([f" {qp} {qe[0]}" if qe[0][0]=="Q" else f" {qp} \"{qe[0]}\"" for qp,qe in stat_obj["qualifiers"]])
                    o.write(templ.format(stat_obj["subject"], stat_obj["predicate"], stat_obj["object"] if stat_obj["object"][0][0]=="Q" else f"\"{stat_obj['object']}\"", qual_string))
            pickle.dump(to_dump, open(to_pkl, "wb+"))


    except (FileNotFoundError, IOError):
        print("Files not found, check if you cloned the dataset")

def parse_wikipeople_flat_quints():
    """
        Assuming there is a path ./data/WikiPeople
        :return: train/val/test in RDF* where complex statements of multiple qualifiers are split into separate quints
    """
    templ = """ << {0!s} {1!s} {2!s} >> {3!s} {4!s} . \n"""
    SOURCE_DIR = Path("./data/WikiPeople")
    try:
        files = list(SOURCE_DIR.glob("*.json"))
        print(files)
        for f in files:
            outputname = str(f).split(".json")[0] + "_flat_quints.rs"
            to_pkl = str(f).split(".json")[0] + "_flat_quints.pkl"
            to_dump = []
            with open(str(f), "r") as sourcef, open(outputname, "w") as o:
                for line in sourcef:
                    stat_obj = {}
                    stat_obj["qualifiers"] = []
                    statement = json.loads(line)
                    for k in list(statement.keys()):
                        if "_h" in k:
                            stat_obj["subject"] = statement[k]
                            stat_obj["predicate"] = k.split("_")[0]
                        elif '_t' in k:
                            stat_obj["object"] = statement[k]
                        elif k != "N":
                            stat_obj["qualifiers"].append((k, statement[k]))

                    if len(stat_obj["qualifiers"]) == 0:
                        to_dump.append(Quint(s=stat_obj['object'], p=stat_obj["predicate"], o=stat_obj["object"] if stat_obj["object"][0][0] == "Q" else f"\"{stat_obj['object']}\"", qp=None, qe=None))
                        o.write(templ.format(stat_obj["subject"], stat_obj["predicate"], stat_obj["object"] if stat_obj["object"][0][0] == "Q" else f"\"{stat_obj['object']}\"", "", ""))
                    else:
                        # create flat quints from the object
                        for qp,qe in stat_obj["qualifiers"]:
                            main_obj = stat_obj["object"] if stat_obj["object"][0][0] == "Q" else f"\"{stat_obj['object']}\""
                            qual_o = qe[0] if qe[0][0]=="Q" else f"\"{qe[0]}\""
                            o.write(templ.format(stat_obj["subject"],
                                                 stat_obj["predicate"],
                                                 main_obj,
                                                 qp, qual_o))
                            to_dump.append(Quint(s=stat_obj['object'], p=stat_obj["predicate"], o=main_obj, qp=qp, qe=qual_o))
            pickle.dump(to_dump, open(to_pkl, "wb+"))

    except (FileNotFoundError, IOError):
        print("Files not found, check if you cloned the dataset")

def parse_wikipeople_std_reif():
    """
        Assuming there is a path ./data/WikiPeople
        :return: train/val/test in N-Triples with standard reification scheme a-la DBpedia 2018
    """
    templ = """ {0!s} {1!s} {2!s} . \n"""
    SOURCE_DIR = Path("./data/WikiPeople")
    try:
        files = list(SOURCE_DIR.glob("*.json"))
        print(files)
        for f in files:
            outputname = str(f).split(".json")[0] + "_stdreif.ttl"
            to_pkl = str(f).split(".json")[0] + "_stdreif.pkl"
            to_dump = []
            with open(str(f), "r") as sourcef, open(outputname, "w") as o:
                for i, line in enumerate(sourcef):
                    stat_obj = {}
                    stat_obj['sid'] = f"ex:sid{i}"
                    stat_obj["qualifiers"] = []
                    statement = json.loads(line)
                    for k in list(statement.keys()):
                        if "_h" in k:
                            stat_obj["subject"] = statement[k]
                            stat_obj["predicate"] = k.split("_")[0]
                        elif '_t' in k:
                            stat_obj["object"] = statement[k]
                        elif k != "N":
                            stat_obj["qualifiers"].append((k, statement[k]))

                    o.write(templ.format(stat_obj['sid'], "rdf:subject", stat_obj['subject']))
                    o.write(templ.format(stat_obj['sid'], "rdf:predicate", stat_obj['predicate']))
                    o.write(templ.format(stat_obj['sid'], "rdf:object", stat_obj['object'] if stat_obj["object"][0][0] == "Q" else f"\"{stat_obj['object']}\""))

                    to_dump.append([stat_obj['sid'], "rdf:subject", stat_obj['subject']])
                    to_dump.append([stat_obj['sid'], "rdf:predicate", stat_obj['predicate']])
                    to_dump.append([stat_obj['sid'], "rdf:object", stat_obj['object'] if stat_obj["object"][0][0] == "Q" else f"\"{stat_obj['object']}\""])

                    for qp,qe in stat_obj["qualifiers"]:
                        qual_v = qe[0] if qe[0][0]=="Q" else f"\"{qe[0]}\""
                        o.write(templ.format(stat_obj['sid'], qp, qual_v))
                        to_dump.append([stat_obj['sid'], qp, qual_v])
            pickle.dump(to_dump, open(to_pkl, "wb+"))

    except (FileNotFoundError, IOError):
        print("Files not found, check if you cloned the dataset")

if __name__ == "__main__":
    parse_wikipeople()
    parse_wikipeople_flat_quints()
    parse_wikipeople_std_reif()