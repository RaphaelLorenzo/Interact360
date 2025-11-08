# debugging tool because we had a problem of labelling the last episodes of the following recordings:
import os
import argparse
import sys
from tqdm import tqdm
import numpy as np
import socket
import pickle

here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '../../'))
sys.path.append(os.path.join(here))

from misc.print_utils import *
from misc.other_utils import *

RECORDINGS_TO_CORRECT = [
    "2022_09_21_astor_place_landfill",
    "2022_09_21_astor_place_recycle",
    "2022_09_26_astor_place_landfill",
    "2022_09_26_astor_place_recycle",
    "2022_09_28_astor_place_landfill",
    "2022_09_28_astor_place_recycle",
    "2022_10_06_astor_place_landfill",
    "2022_10_06_astor_place_recycle",
    "2022_10_12_astor_place_landfill_0",
    "2022_10_12_astor_place_landfill_1",
    "2022_10_12_astor_place_recycle_0",
    "2022_10_12_astor_place_recycle_1",
    "2023_07_06_albee_square_landfill_0",
    "2023_07_06_albee_square_landfill_1",
    "2023_07_06_albee_square_recycle_0",
    "2023_07_06_albee_square_recycle_1",
    "2023_07_07_albee_square_landfill_0",
    "2023_07_07_albee_square_landfill_1",
    "2023_07_07_albee_square_recycle_0",
    "2023_07_07_albee_square_recycle_1",
    "2023_07_11_albee_square_landfill_0",
    "2023_07_11_albee_square_landfill_1",
    "2023_07_11_albee_square_recycle_0",
    "2023_07_11_albee_square_recycle_1",
]

STORAGE_DIR = "/media/raphael/hucedisk3/SSUP/FullVideos/extracted"



for recording in RECORDINGS_TO_CORRECT:
    recording_dir = os.path.join(STORAGE_DIR, recording)
    
    episodes_flags_path = os.path.join(recording_dir, "episodes_flag.json")
    episodes_flags_dict = read_json_to_dic(episodes_flags_path)
    episode_ids = list(episodes_flags_dict.keys())
    episode_ids.sort()
    last_episode_id = episode_ids[-1]
    
    episodes_flags_dict[last_episode_id]["flag"] = "invalid"
    write_dic_to_json_file(episodes_flags_dict, episodes_flags_path)
    prSuccess(f"Assigned invalid label to last episode of {recording}: {last_episode_id}")
    

