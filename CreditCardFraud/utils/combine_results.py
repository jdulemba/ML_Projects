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

if not os.path.isfile(input_files[0]): raise ValueError(f"Could not find {input_files[0]}")
output_dict = pickle.load(open(input_files[0], "rb"))

for idx in tqdm(range(1, len(input_files))):
    try:
        tmp_dict = pickle.load(open(input_files[idx], "rb"))

        for key in output_dict.keys():
            if key == "MetaData": assert output_dict[key] == tmp_dict[key], "Only files using the same data setup (MetaData) should be combined right now."
            if output_dict[key] != tmp_dict[key]: output_dict[key].update(tmp_dict[key])
    except:
        raise ValueError("File %s (number %i) could not be added" % (input_files[idx], idx))


with open(outfname, "wb") as outfile:
    pickle.dump(output_dict, outfile)
print(f"{outfname} written")
