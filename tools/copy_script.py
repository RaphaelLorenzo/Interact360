import os
import shutil

# 1. List all files in a directory
source_dir = '/home/raphael/group_storage/rlorenzo/SSUP/FullVideos/extracted/2022_09_26_astor_place_landfill/episodes/0005/images_360/'
files = os.listdir(source_dir)

# Optional: filter only files (ignore subdirectories)
files = [f for f in files if os.path.isfile(os.path.join(source_dir, f))]
files.sort()

# Print files with their index
for i, file in enumerate(files):
    print(f"{i}: {file}")

# 2. Copy file(s) by index
# Example: copy files 2 through 5 (Python slice is exclusive at the end)
start_index = 2930
end_index = 3030

destination_dir = '/home/raphael/Projects/github/Interact360/demo/data/SSUP_SAMPLE_1_data/'
os.makedirs(destination_dir, exist_ok=True)

for file in files[start_index:end_index]:
    src = os.path.join(source_dir, file)
    dst = os.path.join(destination_dir, file)
    shutil.copy2(src, dst)  # copy2 preserves metadata

print(f"Copied {end_index - start_index} files from {source_dir} to {destination_dir}")
