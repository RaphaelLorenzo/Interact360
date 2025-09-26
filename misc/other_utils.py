
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import json

def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 3).astype(np.float32) / 255.0

def write_dic_to_json_file(dic, file):
    assert(file.endswith(".json"))
    with open(file, "w") as outfile:
        json.dump(dic, outfile)
        
def read_json_to_dic(file: str) -> dict:
    """ Create dictionnary from a json path

    Args:
        file (str): path to read from

    Returns:
        dict: output dic with json lib
    """
    assert(file.endswith(".json"))
    with open(file) as json_file:
        data = json.load(json_file)
    return data