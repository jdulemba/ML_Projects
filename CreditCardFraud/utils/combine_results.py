from pdb import set_trace


from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("output_fname", type=str, help="Name of output file with file extension.")
parser.add_argument("input_files", type=str, help="Input files separated by ':'")
args = parser.parse_args()

input_files = args.input_files.split(":")

outfname = args.output_fname
if not outfname.endswith(".pkl"): outfname = outfname+".pkl"
import pandas as pd
import numpy as np
import os

# open file which has traiing results
import pickle
from tqdm import tqdm

def add(obj1, obj2):
    if obj1 == obj2: return obj1
    assert isinstance(obj1, type(obj2)), "Objects must be the same type in order to add them together!"
    if isinstance(obj1, dict):
        output = {}
        for key in list(set(list(obj1.keys())+list(obj2.keys()))):
            if obj1[key] == obj2[key]:
                output[key] = obj1[key]
            else:
                output[key] = add(obj1[key], obj2[key])
    elif isinstance(obj1, list):
        output = sorted(set(obj1 + obj2))
    else:
            raise ValueError("Only dictionary or list objects can be added with this function")

    return output

if not os.path.isfile(input_files[0]): raise ValueError(f"Could not find {input_files[0]}")
output_dict = pickle.load(open(input_files[0], "rb"))

for idx in tqdm(range(1, len(input_files))):
    try:
        tmp_dict = pickle.load(open(input_files[idx], "rb"))

        for key in output_dict.keys():
            if (key == "MetaData") and (output_dict[key] != tmp_dict[key]):
                merged_dicts = add(tmp_dict[key], output_dict[key])
                output_dict[key].update(merged_dicts)
            else:
                output_dict[key].update(tmp_dict[key])
    except:
        raise ValueError("File %s (number %i) could not be added" % (input_files[idx], idx))

with open(outfname, "wb") as outfile:
    pickle.dump(output_dict, outfile)
print(f"{outfname} written")
