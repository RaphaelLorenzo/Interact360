import argparse
import os
import sys
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '../'))
from misc.print_utils import *
from misc.other_utils import *

def get_image_timestamp(image_file: str) -> datetime.datetime:
    """ Get the timestamp of the image file

    Args:
        image_file (str): path to the image file (basename or full path)

    Returns:
        datetime.datetime: timestamp of the image file
    """
    image_bn = os.path.basename(image_file).split(".")[0]
    image_ts = datetime.datetime.strptime(image_bn, "%Y_%m_%d_%H_%M_%S_%f")

    return image_ts


def is_episode_usable(args, episode_flag):
    if "invalid" in episode_flag:
        # do not consider this episode
        return False
    
    flags = episode_flag.split(",")
    flags = [flag.strip() for flag in flags]
    flags.sort()
    
    if flags == ["checked", "valid"] or flags == ["checked", "eng_360", "valid"] or flags == ["checked", "eng_main", "valid"]:
        return True
    else:
        return False


def get_det_quality(det : dict, kp_score_threshold : float = 0.6) -> float:
    """ Get the quality of a detection based on the ViTPose scores (percentage of keypoints with a score >= kp_score_threshold)

    Args:
        det (dict): detection dictionary
        kp_score_threshold (float): threshold for the keypoints scores

    Returns:
        float: quality of the detection
    """
    vitpose_score = np.array(det["scores_vitpose"])
    det_quality = (vitpose_score >= kp_score_threshold).sum() / len(vitpose_score)
    return det_quality


def get_contiguous_true_segments(array: np.array) -> list:
    """ Get the contiguous true segments of a boolean array
    
    Example : [False,False,True,True,False,True,True,True] returns [(2, 2), (5, 3)]
    
    Args:
        array (np.array): boolean array
        
    Returns:
        list: list of tuples (start, length) with the positions of the true segments and their lengths. Inclusive of the first element.
    """
    
    assert(array.dtype == bool), "array must be a boolean array"
    assert(array.ndim == 1), "array must be a 1D array"
    assert(np.sum(array) > 0), "array must have at least one True"
    #Indices are found here
    segment_begin_positions = np.argwhere(np.diff(array,prepend=False,append=False)) # whenever it shifts, we add Falses, at both ends, since we are searching for Trues
    assert(len(segment_begin_positions) % 2 == 0), "array should always yield an even number of segment begin positions"
    #Conversion into list of 2-tuples
    segments_from_to = segment_begin_positions.reshape(len(segment_begin_positions)//2,2)
    segments_from_to = [tuple(r) for r in segments_from_to] # start position to start of the next segment (ie a segment of Falses)
    segments_from_length = [(start, to - start) for (start, to) in segments_from_to]
    return segments_from_length

def get_tracks_infos(detections_dict: list, engagement_key: str = "engagement", allow_skip: bool = False) -> dict:
    """ Get the infos of the tracks in the detections dictionary

    Args:
        detections_dict (list): detections dictionary (actually a list in which each element is associated with one image input)
        engagement_key (str): key of the engagement in the detections dictionary
        allow_skip (bool): if True, allow to handle errors (these are episodes that are not valid anyway)

    Returns:
        dict: tracks infos ({track_id : {images_indexes : list, validities : list, qualities : list}, ...})
    """
    
    # deeper analysis
    tracks_infos = {}
    # check that the images are valid
    for image_idx, image_element in enumerate(detections_dict):
        for det_idx, det in enumerate(image_element["detections"]):
            associated_track_id = det.get("associated_track_id", -1)
            track_id = associated_track_id if associated_track_id >= 0 else det["track_id"]
            assert("validity" in det.keys()), "Validity not found in detection (got : {})".format(det.keys())
            validity = det["validity"]
            
            if validity == "valid":
                engagement = det.get(engagement_key, -1)
            else:
                engagement = -1
                
            if track_id not in tracks_infos:
                tracks_infos[track_id] = {
                    "images_indexes": [],
                    "det_indexes": [],
                    "validities": [],
                    "qualities": [],
                    "engagements": [],
                }
            tracks_infos[track_id]["images_indexes"].append(image_idx)
            tracks_infos[track_id]["det_indexes"].append(det_idx)
            tracks_infos[track_id]["validities"].append(validity)
            try:
                quality = get_det_quality(det)
            except Exception as e:
                if allow_skip:
                    quality = 1.0
                else:
                    raise e
            tracks_infos[track_id]["qualities"].append(quality)
            tracks_infos[track_id]["engagements"].append(engagement)
        
    return tracks_infos

def get_image_shape(detections_dict: list) -> tuple:
    """ Get the shape of the image from the detections dictionary

    Args:
        detections_dict (list): detections dictionary (actually a list in which each element is associated with one image input)

    Returns:
        tuple: shape of the image (height, width)
    """
    for image_element in detections_dict:
        # they should all have the same shape anyway
        image_shape = image_element["image_shape"]
        return image_shape
    return None


RECORDING_TO_PLACE = {
    "rosbag2_2025_07_07-10_24_20":	"EntranceCBack-1",
    "rosbag2_2025_07_07-10_49_31":	"Cafeteria-1",
    "rosbag2_2025_07_07-11_16_10":	"Room104-1",
    "rosbag2_2025_07_07-12_38_45":	"Room104-1",
    "rosbag2_2025_07_07-15_33_32":	"Room104-1",
    "rosbag2_2025_07_10-10_29_13":	"Room005-1",
    "rosbag2_2025_07_10-15_47_18":	"Room005-1",
    "rosbag2_2025_07_11-10_28_07":	"MainEntrance-1",
    "rosbag2_2025_07_11-11_18_00":	"MainEntrance-1",
    "rosbag2_2025_07_11-13_27_26":	"CoffeeB-1",
    "rosbag2_2025_07_11-14_54_55":	"CoffeeB-1",
    "rosbag2_2025_07_15-12_39_21":	"Room104-2",
    "rosbag2_2025_07_15-13_41_01":	"Room104-2",
    "rosbag2_2025_07_15-14_48_22":	"Room104-2",
    "rosbag2_2025_07_16-13_12_03":	"EntranceCBack-2",
    "rosbag2_2025_07_16-14_07_49":	"EntranceCBack-2",
    "rosbag2_2025_07_16-15_50_45":	"EntranceCBack-2",
    "rosbag2_2025_07_17-11_28_34":	"CoffeeB-2",
    "rosbag2_2025_07_17-12_52_12":	"CoffeeB-2",
    "rosbag2_2025_07_18-10_37_07":	"MainEntrance-2",
    "rosbag2_2025_07_21-10_22_11":	"Room104-3",
    "rosbag2_2025_07_21-11_56_40":	"Room104-3",
    "rosbag2_2025_07_21-13_09_22":	"Room104-3",
    "rosbag2_2025_07_21-14_11_37":	"Room104-3",
    "rosbag2_2025_07_21-15_15_07":	"Room104-3",
    "rosbag2_2025_07_22-09_38_18":	"Cafeteria-2",
    "rosbag2_2025_07_22-10_59_25":	"Cafeteria-2",
    "rosbag2_2025_07_22-12_18_30":	"Cafeteria-2",
    "rosbag2_2025_07_22-13_30_39":	"Cafeteria-2",
    "rosbag2_2025_07_23-11_01_56":	"EntranceCBack-3",
    "rosbag2_2025_07_23-12_18_45":	"EntranceCBack-3",
    "rosbag2_2025_07_23-13_17_40":	"EntranceCBack-3",
    "rosbag2_2025_07_23-14_30_55":	"EntranceCBack-3",
    "rosbag2_2025_07_24-10_41_01":	"CoffeeB-3",
    "rosbag2_2025_07_24-12_14_56":	"CoffeeB-3",
    "rosbag2_2025_07_24-13_33_54":	"CoffeeB-3",
    "rosbag2_2025_07_24-14_33_36":	"CoffeeB-3",
    "rosbag2_2025_07_25-10_52_32":	"MainEntrance-3",
    "rosbag2_2025_07_25-14_14_16":	"MainEntrance-3",
    "rosbag2_2025_07_28-10_18_10":	"Room104-4",
    "rosbag2_2025_07_28-11_25_33":	"Room104-4",
    "rosbag2_2025_07_28-13_05_46":	"Room104-4",
    "rosbag2_2025_07_28-14_19_07":	"Room104-4",
    "rosbag2_2025_07_29-10_23_14":	"EntranceCBack-4",
    "rosbag2_2025_07_29-13_17_18":	"EntranceCBack-4",
    "rosbag2_2025_07_29-14_09_46":	"EntranceCBack-4",
    "rosbag2_2025_10_07-15_03_48":	"EntranceCFacing-1",
    "rosbag2_2025_10_07-16_21_39":	"EntranceCFacing-1",
    "rosbag2_2025_10_09-08_52_20":	"MainEntrance-4",
    "rosbag2_2025_10_09-10_23_38":	"MainHallway-1",
    "rosbag2_2025_10_09-17_37_23":	"EntranceCFacing-2",
    "rosbag2_2025_10_09-18_50_21":	"EntranceCFacing-2",
    "rosbag2_2025_10_15-09_37_49":	"MainHallway-2",
    "rosbag2_2025_10_15-11_03_05":	"MainHallway-2",
    "rosbag2_2025_10_15-12_02_00":	"EntranceCBack-5",
    "rosbag2_2025_10_15-12_27_14":	"EntranceCFacing-3",
    "rosbag2_2025_10_15-13_11_27":	"EntranceCBack-6",
    "rosbag2_2025_10_15-14_02_29":	"EntranceCBack-6",
    "rosbag2_2025_10_15-14_30_03":	"EntranceCBack-6",
    "rosbag2_2025_10_16-09_27_48":	"EntranceCBack-7",
    "rosbag2_2025_10_16-11_29_56":	"EntranceCFacing-5",
    "rosbag2_2025_10_16-12_47_57":	"EntranceCBack-8",
    "rosbag2_2025_10_17-13_19_29":	"Room104-5",
    "rosbag2_2025_10_17-14_28_15":	"Room104-5",
    "rosbag2_2025_10_17-15_11_00":	"Room104-5",
    "rosbag2_2025_10_17-16_47_09":	"Room104-5",
    "rosbag2_2025_10_20-10_13_32":	"Bulle12X-1",
    "rosbag2_2025_10_20-11_51_02":	"Bulle12X-1",
    "rosbag2_2025_10_20-16_02_46":	"Bulle12X-1",
    "2022_09_21_astor_place_landfill":"AstorPlace-Moving",
    "2022_09_21_astor_place_recycle":"AstorPlace-Moving",
    "2022_09_26_astor_place_landfill":"AstorPlace-Moving",
    "2022_09_26_astor_place_recycle":"AstorPlace-Moving",
    "2022_09_28_astor_place_landfill":"AstorPlace-Moving",
    "2022_09_28_astor_place_recycle":"AstorPlace-Moving",
    "2022_10_06_astor_place_landfill":"AstorPlace-Moving",
    "2022_10_06_astor_place_recycle":"AstorPlace-Moving",
    "2022_10_12_astor_place_landfill_0":"AstorPlace-Moving",
    "2022_10_12_astor_place_landfill_1":"AstorPlace-Moving",
    "2022_10_12_astor_place_recycle_0":"AstorPlace-Moving",
    "2022_10_12_astor_place_recycle_1":"AstorPlace-Moving",
    "2023_07_06_albee_square_landfill_0":"AlbeeSquare-Moving",
    "2023_07_06_albee_square_landfill_1":"AlbeeSquare-Moving",
    "2023_07_06_albee_square_recycle_0":"AlbeeSquare-Moving",
    "2023_07_06_albee_square_recycle_1":"AlbeeSquare-Moving",
    "2023_07_07_albee_square_landfill_0":"AlbeeSquare-Moving",
    "2023_07_07_albee_square_landfill_1":"AlbeeSquare-Moving",
    "2023_07_07_albee_square_recycle_0":"AlbeeSquare-Moving",
    "2023_07_07_albee_square_recycle_1":"AlbeeSquare-Moving",
    "2023_07_11_albee_square_landfill_0":"AlbeeSquare-Moving",
    "2023_07_11_albee_square_landfill_1":"AlbeeSquare-Moving",
    "2023_07_11_albee_square_recycle_0":"AlbeeSquare-Moving",
    "2023_07_11_albee_square_recycle_1":"AlbeeSquare-Moving",
    "2023_07_12_albee_square_landfill_0":"AlbeeSquare-Moving",
    "2023_07_12_albee_square_landfill_1":"AlbeeSquare-Moving",
    "2023_07_12_albee_square_recycle_0":"AlbeeSquare-Moving",
    "2023_07_12_albee_square_recycle_1":"AlbeeSquare-Moving",
    "2023_07_14_albee_square_landfill":"AlbeeSquare-Moving",
    "2023_07_14_albee_square_recycle":"AlbeeSquare-Moving",
}