from pdb import set_trace
import os
import fnmatch
import pickle
from tqdm import tqdm

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("jobdir", help="The job directory name used as the output directory for this and the subsequent script's output.")
parser.add_argument("res_type", choices=["Train", "Test"], help="Choose to plot results from model training or testing.")
args = parser.parse_args()

dirname = os.path.join(os.environ["RESULTS_DIR"], args.jobdir)
if not os.path.isdir(dirname): raise ValueError(f"Directory {dirname} not found")

input_files = [os.path.join(dirname, "indiv_model_output", fname) for fname in fnmatch.filter(os.listdir(os.path.join(dirname, "indiv_model_output")), f"*{args.res_type}*ModelInfo*")]
outfname = os.path.join(dirname, f"{args.res_type}ingResults.pkl")

if not os.path.isfile(input_files[0]): raise ValueError(f"Could not find {input_files[0]}")
output_dict = pickle.load(open(input_files[0], "rb"))

for idx in tqdm(range(1, len(input_files))):
    try:
        tmp_dict = pickle.load(open(input_files[idx], "rb"))

        for key in output_dict.keys():
            try:
                output_dict[key].update(tmp_dict[key])
            except:
                print(f"{key} could not be added to file")
    except:
        raise ValueError("File %s (number %i) could not be added" % (input_files[idx], idx))

with open(outfname, "wb") as outfile:
    pickle.dump(output_dict, outfile)
print(f"{outfname} written")
