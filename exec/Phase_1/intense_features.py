# intensifier featuture extraction 

import numpy as np
import intensifiers

def get_strong_affs(x):
    safs = []
    for s in x:
        count = 0
        for word in list(s):
            if word in intensifiers.strong_affirmatives: count += 1
        safs.append(count)
    return np.array(safs).reshape(-1, 1)

def get_strong_negs(x):
    safs = []
    for s in x:
        count = 0
        for word in list(s):
            if word in intensifiers.strong_negations: count += 1
        safs.append(count)
    return np.array(safs).reshape(-1, 1)

def get_strong_inter(x):
    safs = []
    for s in x:
        count = 0
        for word in list(s):
            if word in intensifiers.interjections: count += 1
        safs.append(count)
    return np.array(safs).reshape(-1, 1)

def get_strong_intense(x):
    safs = []
    for s in x:
        count = 0
        for word in list(s):
            if word in intensifiers.intensifiers: count += 1
        safs.append(count)
    return np.array(safs).reshape(-1, 1)