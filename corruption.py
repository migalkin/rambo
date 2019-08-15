import pickle
from typing import List
import numpy as np
import random
from tqdm import tqdm_notebook as tqdm

from utils import *


def sample_negatives(quint: Quint, probs: List[float]) -> Quint:
    """ probs: [ p(s), p(r), p(o), p(q) ] """
    assert np.sum(probs) == 1.0
    assert len(probs) == 4
    # print(probs)
    l = np.random.choice(["s", "p", "o", "q"], 1, p=probs)
    if l[0] == "s":
        return Quint(s=random.choice(entities), p=quint[1], o=quint[2], qp=quint[3], qe=quint[4])
    #         while True:
    #             new_s = random.choice(entities)
    #             q = Quint(s=new_s, p=quint[1], o=quint[2], qp=quint[3], qe=quint[4])
    #             if q not in raw_data:
    #                 return q
    elif l[0] == "p":
        return Quint(s=quint[0], p=random.choice(predicates), o=quint[2], qp=quint[3], qe=quint[4])
    #         while True:
    #             new_p = random.choice(predicates)
    #             q = Quint(s=quint[0], p=new_p, o=quint[2], qp=quint[3], qe=quint[4])
    #             if q not in raw_data:
    #                 return q
    elif l[0] == "o":
        return Quint(s=quint[0], p=quint[1], o=random.choice(entities), qp=quint[3], qe=quint[4])
    #         while True:
    #             new_o = random.choice(entities)
    #             q = Quint(s=quint[0], p=quint[1], o=new_o, qp=quint[3], qe=quint[4])
    #             if q not in raw_data:
    #                 return q
    elif l[0] == "q":
        if quint[3]:
            if np.random.random() > 0.5:
                # sample qp
                return Quint(s=quint[0], p=quint[1], o=quint[2], qp=random.choice(predicates), qe=quint[4])
            #                 while True:
            #                     qp = random.choice(predicates)
            #                     q = Quint(s=quint[0], p=quint[1], o=quint[2], qp=qp, qe=quint[4])
            #                     if q not in raw_data:
            #                         return q
            else:
                return Quint(s=quint[0], p=quint[1], o=quint[2], qp=quint[3], qe=random.choice(entities))
        #                 while True:
        #                     qe = random.choice(entities)
        #                     q = Quint(s=quint[0], p=quint[1], o=quint[2], qp=quint[3], qe=qe)
        #                     if q not in raw_data:
        #                         return q
        else:
            return Quint(s=quint[0], p=quint[1], o=quint[2], qp=random.choice(predicates), qe=random.choice(entities))


#             while True:
#                 qp = random.choice(predicates)
#                 qe = random.choice(entities)
#                 q = Quint(s=quint[0], p=quint[1], o=quint[2], qp=qp, qe=qe)
#                 if q not in raw_data:
#                     return q

if __name__ == "__main__":
    # Just checking things
    probs = [0.3, 0.0, 0.3, 0.4]
    q_neg = sample_negatives(raw_data[0], probs)
    print(q_neg)

    l = np.random.choice(["s", "p", "o", "q"], 1000, p=probs)
    print(l[0])
    unique, counts = np.unique(l, return_counts=True)
    dict(zip(unique, counts))
    # l.count("s"), l.count("p"), l.count("o"), l.count("q")

    negative_samples = []
    for q in tqdm(raw_data):
        negative_samples.append(sample_negatives(q, probs))

    count = 0
    for n in tqdm(negative_samples):
        if n in raw_data:
            print(n)
            count += 1

    print(f"{count} / {len(raw_data)} are not unique negatives")