import os
import shutil

DIR_LIST = [
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
    "2023_07_12_albee_square_landfill_0",
    "2023_07_12_albee_square_landfill_1",
    "2023_07_12_albee_square_recycle_0",
    "2023_07_12_albee_square_recycle_1",
    "2023_07_14_albee_square_landfill",
    "2023_07_14_albee_square_recycle",
    "2022_09_21_astor_place_landfill",
    "2022_09_21_astor_place_recycle",
]

ORIG_DIR = "/home/raphael/group_storage/rlorenzo/SSUP/FullVideos/extracted"
DEST_DIR = "/media/raphael/hucedisk3/SSUP/FullVideos/extracted"
if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)

for dir in DIR_LIST:
    orig_dir = os.path.join(ORIG_DIR, dir)
    dest_dir = os.path.join(DEST_DIR, dir)
    print("----- Processing directory: {} -----".format(dir))
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        os.makedirs(os.path.join(dest_dir, "episodes"), exist_ok=True)
        
        episodes_desc_path = os.path.join(orig_dir, "episodes.json")
        episodes_flags_path = os.path.join(orig_dir, "episodes_flag.json")
        
        if os.path.exists(episodes_desc_path):
            shutil.copy(episodes_desc_path, os.path.join(dest_dir, "episodes.json"))
        if os.path.exists(episodes_flags_path):
            shutil.copy(episodes_flags_path, os.path.join(dest_dir, "episodes_flag.json"))
            
        list_episodes = os.listdir(os.path.join(orig_dir, "episodes"))
        list_episodes.sort()
        for episode in list_episodes:
            episode_dir = os.path.join(orig_dir, "episodes", episode)
            episode_dest_dir = os.path.join(dest_dir, "episodes", episode)
            if not os.path.exists(episode_dest_dir):
                os.makedirs(episode_dest_dir)
                
            detections_path = os.path.join(episode_dir, "sam2_detections_360.json")
            interact_zone_tracking_path = os.path.join(episode_dir, "interact_zone_tracking_360.json")
            if os.path.exists(detections_path):
                print(f"[{dir}][{episode}] Copying detections from {detections_path} to {os.path.join(episode_dest_dir, 'sam2_detections_360.json')}")
                shutil.copy(detections_path, os.path.join(episode_dest_dir, "sam2_detections_360.json"))
            if os.path.exists(interact_zone_tracking_path):
                print(f"[{dir}][{episode}] Copying interact zone tracking from {interact_zone_tracking_path} to {os.path.join(episode_dest_dir, 'interact_zone_tracking_360.json')}")
                shutil.copy(interact_zone_tracking_path, os.path.join(episode_dest_dir, "interact_zone_tracking_360.json"))
                
        
