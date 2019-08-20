import json
from pathlib import Path

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
            outputname = str(f).split(".json")[0]+".rs"
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
                    qual_string = " ; \n\t ".join([f" {qp} {qe[0]}" if qe[0][0]=="Q" else f" {qp} \"{qe[0]}\"" for qp,qe in stat_obj["qualifiers"]])
                    o.write(templ.format(stat_obj["subject"], stat_obj["predicate"], stat_obj["object"] if stat_obj["object"][0][0]=="Q" else f"\"{stat_obj['object']}\"", qual_string))


    except (FileNotFoundError, IOError):
        print("Files not found, check if you cloned the dataset")

if __name__ == "__main__":
    parse_wikipeople()
