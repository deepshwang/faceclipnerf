#!/bin/bash
set -x
##### [0] Set data directory & variable #####
#PARAMETERS
DATA_DIR="/home/user1/data/hypernerf"
CAPTURE_NAME="shwangv5"
n=9 #sampling rate (flattening)

#video flattening
PROJ_DIR=$DATA_DIR/$CAPTURE_NAME
VIDEO_PATH=$DATA_DIR/$CAPTURE_NAME/$CAPTURE_NAME".mov"
RGB_RAW_DIR=$DATA_DIR/$CAPTURE_NAME/"rgb-raw"
OUT_PATTERN=$RGB_RAW_DIR"/%06d.png"
FILTERS="mpdecimate,setpts=N/FRAME_RATE/TB,scale=iw:ih,yadif"
#Feature matching & SFM recon.
COLMAP_IMAGE_SCALE=${1:-1}
COLMAP_DB_PATH=$PROJ_DIR/"database.db"
COLMAP_RGB_DIR=$PROJ_DIR"/rgb/${COLMAP_IMAGE_SCALE}x"
COLMAP_MASK_DIR=$PROJ_DIR"/mask/${COLMAP_IMAGE_SCALE}x"
SHARE_INTRINSICS=1
ASSUME_UPRIGHT_CAMERAS=1
COLMAP_OUT_PATH=$PROJ_DIR"/sparse"
mkdir $COLMAP_OUT_PATH
refine_principal_point=1 # 1 is True
min_num_matches=32
filter_max_reproj_error=2
tri_complete_max_reproj_error=2

# ##### [1] Flatten video #####
FPS=60/$n
mkdir -p $RGB_RAW_DIR
/usr/bin/ffmpeg -i $VIDEO_PATH -r $FPS -vf $FILTERS $OUT_PATTERN

##### [2] Downsample frames #####
python downsample_frames.py --data_dir $DATA_DIR \
                            --capture_name $CAPTURE_NAME

##### [3] Extract mask #####
python extract_mask.py --data_dir $DATA_DIR \
                       --capture_name $CAPTURE_NAME

##### [4] Extract features #####
colmap feature_extractor \
--SiftExtraction.use_gpu 0 \
--SiftExtraction.upright 1 \
--ImageReader.camera_model OPENCV \
 --ImageReader.single_camera 1 \
 --database_path ${COLMAP_DB_PATH} \
 --ImageReader.mask_path ${COLMAP_MASK_DIR} \
 --image_path ${COLMAP_RGB_DIR}

##### [5] Match features #####
if [ ! -f vocab_tree_flickr100K_words32K.bin ]; then
    wget https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin
fi
colmap vocab_tree_matcher \
    --SiftMatching.use_gpu 0 \
    --VocabTreeMatching.vocab_tree_path vocab_tree_flickr100K_words32K.bin \
    --database_path ${COLMAP_DB_PATH}

##### [6] SFM reconstruction #####
colmap mapper \
  --Mapper.ba_refine_principal_point $refine_principal_point \
  --Mapper.filter_max_reproj_error $filter_max_reproj_error \
  --Mapper.tri_complete_max_reproj_error $tri_complete_max_reproj_error \
  --Mapper.min_num_matches $min_num_matches \
  --database_path $COLMAP_DB_PATH \
  --image_path $COLMAP_RGB_DIR \
  --output_path $COLMAP_OUT_PATH

##### [7] Final parsing of data #####
python parse_data.py --data_dir $PROJ_DIR
