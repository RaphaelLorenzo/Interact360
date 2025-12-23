# Interact360

![concept_tracking](illustrations/concept_tracking.jpg)


## Install
Requirements : CUDA equipped GPU with at least 7GB of VRAM. 3GB of disk space for checkpoints.

- Install `ffmpeg` if you dont have it already (for video extraction)
```
sudo apt-get install ffmpeg
```

- Create a conda environment and install dependencies
```
conda create --name interact360env python=3.10
conda activate interact360env
pip install -r requirements.txt
```

- Download the checkpoints (by default it will download necessary checkpoints for `SAM2.1-Tiny` and `Sapiens-03b`)
```
cd checkpoints
./download_ckpts.sh
cd ../
```

## Run

Place your video under `./demo/data` here we provide `ClubInteract-TCAn.mp4` (not a part of HUI360 or SSUP-A dataset, faces are blurred which may affect the performances)

Extract the images (this script uses ffmpeg)
```
python ./process/extract_images.py -i ./demo/data/ClubInteract-TCAn.mp4 -em time -met 60 -w 1
```

Run the detection and tracking process
```
python ./process/detect_segment_and_track.py -d ./demo/output/ClubInteract-TCAn -sm base -sibs 8

# With display (slower)
python ./process/detect_segment_and_track.py -d ./demo/output/ClubInteract-TCAn -sm base -sibs 1 --disp
```

Detect the interactions using the default mask
```
python ./process/detect_interactions.py -d ./demo/output/ClubInteract-TCAn
```

Extract keypoints with ViTPose (17 keypoints)
```
python ./process/extract_poses_vitpose.py -d ./demo/output/ClubInteract-TCAn
```

Extract keypoints with Sapiens (308 keypoints)
```
python ./process/extract_poses_sapiens.py -d ./demo/output/ClubInteract-TCAn
```

Then you can see the results in the visualizer
```
python visualize.py -d ./demo/output -rn ClubInteract-TCAn
```




## Additional informations

<details>
<summary>Download more checkpoints</summary>

Use `./download_ckpts.sh -c all` to download more sizes of checkpoint

</details>


<details>
<summary>Multiprocessing</summary>

In all scripts `-w` is used to specify the number of workers to use when things are parallelized, by default `-w` is `0` and scripts will use all cores, or half of the cores detected. In ViTPose extraction when multiple episodes are present they are parallelized depending on the available GPU VRAM.

</details>



<details>
<summary>Optimized inference</summary>

The option `-vos` enable the use optimized inference with `torch.compile` for the image encoder in SAM2-1.

The option `-b16` enable the use of `bfloat16` checkpoints for Sapiens pose extraction (only available on some GPUs, and with versions of PyTorch that are compatible with the checkpoints)

</details>




<details>
<summary>Extracting video</summary>

**Episodes duration** : use `-em` to set the episode creation method (either `single` i.e. not episode, or `time` i.e. based on a maximum duration, `detection` to create episodes only when people are in front of the camera is not implemented here), use `-met` to set the maximum episode time in seconds

**Other options** : use `-h` to check them

</details>



<details>
<summary>Detection, segmentation and tracking</summary>

The process is explained in the publication. Many options are related to the filtering, and other filtering threhsolds are set as variable at the beggining of `detect_segment_and_track.py`

- `-sibs` refers to the batch size for SAM2-Image initial segmentation (it refers to the number of persons segmented simultaneously, but anyway the full images are processed one by one)
- In the tracking phase, tracks are processed one by one, and a single image at a time 

</details>




<details>
<summary>Mask for interaction</summary>

**Fixed interaction zone** : By default we use the mask of `assets/default_mask_intersection_zone.png` update it according to your needs if you have a fixed interaction zone. 

**Moving interaction zone** : If you have a moving interaction zone (like in SSUP) you can first track the interaction zone by passing its bouding box in the first image of your sequence as `-izt xmin,ymin,xmax,ymax`, note that `xmax` can be `> width` this is the way to track an object that appears on both sides of the equirectangular image. This will create a file `interaction_zone_tracking_360.json` in each episode, that will be used by default by `detect_interaction.py`.
```
python process/detect_segment_and_track.py -d ./demo/output/SSUP_SAMPLE_0/ -izt 2688,1042,4585,1919
```

Where `-izt 2688,1042,4585,1919` is the bounding box of the interaction zone in the first image then it will run a full pass of tracking to find the masks of the interaction zone. Then compute the convex hull of these masks. So you need a convex interaction zone, also this way of doing is not preferred because less reliable, a fix interaction zone - relatively to the camera - is better.
</details>

<details>
<summary>Manual correction tool</summary>

Documentation TBD

</details>

<details>
<summary>Format and structure</summary>

1. When using `extract_images.py` the images will be extracted as episodes in the specified `--save_dir` (by default `./demo/output/`) in a subdirectory (name determined by the filename of the original video). For example `python ./process/extract_images.py -i ./demo/data/MyPanoVideoName.mp4 -em time -met 60 -w 1 --save_dir /media/jdoe/interactpano/` :


```
/media/jdoe/interactpano/MyPanoVideoName
.
├── episodes
│   ├── 0000
│   │   └── images_360
│   └── 0001
│       └── images_360
├── episodes.json
└── metadata_360_extraction.txt
```


A metadata file `episodes.json` will be created and placed at the root of the extraction along with the episodes subdirectory containing metadata about the episode start and end timestamp. It is a list of length equal to the number of extracted episodes.

```
[
    {
        "session": "MyPanoVideoName",
        "episode": "0000",
        "start_time": 1763542047.679756,
        "end_time": 1763542064.8130896
    },
    {
        "session": "MyPanoVideoName",
        "episode": "0001",
        "start_time": ...,
        "end_time": ...
    },
    ...
]
```


2. When using the different steps of processing you will end up with json files, the main one is `annotations_360.json` (if you are using `-izt` option for mobile interaction zone, you will also have `interaction_zone_tracking_360.json`). Along the process copies are created you may remove them later to save space.

```
/media/jdoe/interactpano/MyPanoVideoName
.
├── episodes
│   ├── 0000
│   │   ├── images_360
│   │   ├── annotations_360_copy_2025_10_23_16_46_28.json
│   │   └── annotations_360.json
│   └── 0001
│       ├── images_360
│       ├── annotations_360_copy_2025_10_23_16_40_19.json
│       └── annotations_360.json
├── episodes.json
└── metadata_360_extraction.txt
```

The structure of the `annotations_360.json` file is the following (after pose extraction). It is a list of length of the number of image files in `images_360` for each image a dict contains some metadata : `"image_file"` (just the filename, that should be its formated timestamp) and `"image_shape"` (`height, width, channels`). For each file a list of `"detections"` (length is the number of person detected and tracked in the image), each detection is dict with the following informations. 
- `"track_id"` is assigned to each person tracked along the episode.
- The bounding box coordinates are expressed as `xmin`, `ymin`, `xmax` and `ymax` in pixels. As the 360 image is wrapped, some may exceed the image size (but should not be negative) : pixel `(4283,783)` corresponds to pixel `(443,783)`.
- `"validity"` flag is assigned (default is `valid`) but it might be corrected in the manual correction tool
- `"mask_size"` indicates the total number of pixels in the mask
- `"mask_rle"` is the RLE encoding of the mask as a string of comma separated values, it can be decoded using `decode_RLE` from `misc.rle_tools.py` (using torch tensor format) : `mask = decode_RLE(torch.from_numpy(np.array(det["mask_rle"].split(",")).astype(np.int32)).to(DEVICE), shape = (width, height))`
- `"engagement_level"` is automatically determined by the interaction detection process and can be corrected from the manual correction tool. `0` corresponds to no engagement, `3` corresponds to interaction (currently happening).
- `"intersection_pixels"` is the number of pixels intersecting between the person mask and the interesection zone mask (either fixed or mobile). NB : in older versions this was not exported so `"intersection_pixels" : -1` indicates that this information was not saved.
- `"scores_sapiens_308_03b"` or `"scores_sapiens_308_06b"` is a list of float with lenght 308 that are the confidence scores associated with each keypoint of `"keypoints_sapiens_308_03b"` 
- `"scores_vitpose"` is a list of float with lenght 17 that are the confidence scores associated with each keypoint of `"keypoints_vitpose"` 
- `"keypoints_sapiens_308_03b"` or `"keypoints_sapiens_308_06b"` are the keypoints coordinates (in pixels) with the Goliath format of Sapiens, some were extracted using the 03b pose model and others with 06b pose model. 
- `"keypoints_vitpose"` are the keypoints coordinates (in pixels) with the COCO-17 format of ViTPose.
The keypoints are expressed as lists of length `308*2` or `17*2` (alternating `x` and `y` coordinates), **note that they are expressed as the pixel position of the joint in the image, so when the box is wrapped around they may end up not being in the box numerically, or showing two parts of the body at the two ends of the image**. 
If a keypoint is not present it means that these keypoints were not extracted, probably because the track was already labelled as invalid or the box is too small.

```
[
  {
    "image_file": "1970_01_01_02_00_00_000000.jpg",
    "image_shape": [
        1920,
        3840,
        3
    ]
    "detections": [
      {
        "track_id": 0,
        "xmin": 4283,
        "ymin": 783,
        "xmax": 4451,
        "ymax": 1220,
        "validity": "valid",
        "mask_size": 44302,
        "mask_rle": "0,3007221,17,3820,...,13,2687454",
        "engagement_level": 3,
        "intersection_pixels": 18545,
        "keypoints_vitpose": [
            481,
            812,
            ...
        ],
        "scores_vitpose": [
            0.53,
            ...
        ],
        "keypoints_sapiens_308_03b": [
            498,
            816,
            ...
        ],
        "scores_sapiens_308_03b": [
            0.45,
            ...
        ]
      },
      {
        "track_id": 4,
        "xmin": 1253,
        "ymin": 721,
        "xmax": 1611,
        "ymax": 1520,
        "validity": "invalid",
        "mask_size": 145304,
        "mask_rle": "0,8494527,21,4244,...,14,1425546",
        "engagement_level": 0,
        "intersection_pixels": 0,
        "keypoints_vitpose": [
            1481,
            812,
            ...
        ],
        "scores_vitpose": [
            0.43,
            ...
        ],
        "keypoints_sapiens_308_06b": [
            1498,
            816,
            ...
        ],
        "scores_sapiens_308_06b": [
            0.64,
            ...
        ]
      },
    ]
  },
  {
    "image_file": "1970_01_01_02_00_00_066667.jpg",
    "image_shape": [
        1920,
        3840,
        3
    ]
    "detections": [
      ...
    ]
  },
  ...
]
```

<details>
<summary>Exporting to HUI360 Format</summary>

How to export to the csv format of HUI360
TBD

</details>

<details>
<summary>Exporting to HUI360-Video Format</summary>

How to export (restructure) to the HUI360-Video Format

The original structure is separated, videos are re-created for easier downloading :

```
HUI360-Video/
├── annotations
│   ├── MyFirstPanoVideoName
│   │   └── episodes
│   │       ├── 0000
│   │       │   ├── annotations_360.json
│   │       │   ├── images_360_filenames.txt
│   │       │   └── interaction_zone_tracking_360.json
|   |       ...  
│   │       └── 0001
│   │           ├── ...
│   │           └── ...
│   ...
│   └── AnotherPanoVideoName
│       └── episodes
│           ├── ...
│           └── ...
├── dataset
│   ├── MyFirstPanoVideoName
│   │   ├── episodes
│   │   │   ├── 0000
│   │   │   │   └── images_360
│   │   │   │       ├── ...
│   │   │   │       ├── ...
│   │   │   │       ├── ...
│   │   │   │       └── ...
|   |   |   ...  
│   │   │   └── 0001
│   │   │       └── ...
|   │   ├── episodes_flag.json
│   │   └── episodes.json
│   ...
│   └── AnotherPanoVideoName
│       ├── episodes
│       ...
|       ├── episodes_flag.json
│       └── episodes.json
├── README.md
├── video_full
│   ├── MyFirstPanoVideoName
│   │   └── episodes
│   │       ├── 0000
│   │       │   └── images_360_full.mp4
|   |       ...  
│   │       └── 0001
│   │           ├── ...
│   │           └── ...
│   ...
│   └── AnotherPanoVideoName
│       └── episodes
│           ├── ...
│           └── ...
└── video_mini
    ├── MyFirstPanoVideoName
    │   └── episodes
    │       ├── 0000
    │       │   └── images_360_mini.mp4
    |       ...  
    │       └── 0001
    │           ├── ...
    │           └── ...
    ...
    └── AnotherPanoVideoName
        └── episodes
            ├── ...
            └── ...
```


TBD

</details>


3. Manual correction and visualization tool (`visualize.py`) will create a dictionnary indexed by the episode name with the flags of the episodes (valid by default but can be modified in the tool), 

```
{
    "0000": {
        "start_time": 1756981067.0,
        "end_time": 1756981366.9999995,
        "start_time_formatted": "2025-09-04 12:17:47",
        "end_time_formatted": "2025-09-04 12:22:47",
        "flag": "valid,checked"
    },
    "0001": {
        "start_time": 1756981367.0,
        "end_time": 1756981666.9999995,
        "start_time_formatted": "2025-09-04 12:22:47",
        "end_time_formatted": "2025-09-04 12:27:47",
        "flag": "valid,checked"
    },
    ...
}
```

The tools will also create backups when modifying the tracks for example. You will end up with a structure like this :

```
/media/jdoe/interactpano/MyPanoVideoName
.
├── episodes
│   ├── 0000
│   │   ├── images_360
│   │   ├── annotations_360_copy_2025_10_23_16_46_28.json
│   │   ├── annotations_360_backup_20251023_185128.json
│   │   └── annotations_360.json
│   └── 0001
│       ├── images_360
│       ├── annotations_360_copy_2025_10_23_16_40_19.json
│       ├── annotations_360_backup_20251023_193221.json
│       └── annotations_360.json
├── episodes.json
├── episodes_flag.json
└── metadata_360_extraction.txt
```

</details>



## Acknowledgement 
- SAM2
```
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```

- SAM2 Real Time : 
```
https://github.com/Gy920/segment-anything-2-real-time
```

- ViTPose
```
@inproceedings{
  xu2022vitpose,
  title={Vi{TP}ose: Simple Vision Transformer Baselines for Human Pose Estimation},
  author={Yufei Xu and Jing Zhang and Qiming Zhang and Dacheng Tao},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022},
}
```

- Sapiens
```
@article{khirodkar2024sapiens,
  title={Sapiens: Foundation for Human Vision Models},
  author={Khirodkar, Rawal and Bagautdinov, Timur and Martinez, Julieta and Zhaoen, Su and James, Austin and Selednik, Peter and Anderson, Stuart and Saito, Shunsuke},
  journal={arXiv preprint arXiv:2408.12569},
  year={2024}
}
```


