## Dataset 
* First download the official annotations. Here, we keep the same setting as ego4d official repo and assume that ego4d data are stored under '~/ego4d_data'.
* Run the following command to process the official annotations to our format.
```shell
cd data/ego4d/annotations/
cp ~/ego4d_data/ego4d.json ~/ego4d_data/v1/annotations
python process_annotations.py -i ~/ego4d_data/v1/annotations/
cd ..
```
* Run the following command to split the official video-level features to clip-level features.
```shell
 python prepare_ego4d_dataset.py \
    --input_train_split ~/ego4d_data/v1/annotations/nlq_train.json \
    --input_val_split ~/ego4d_data/v1/annotations/nlq_val.json \
    --input_test_split ~/ego4d_data/v1/annotations/nlq_test_unannotated.json \
    --video_feature_read_path ~/ego4d_data/v1/omnivore_video_swinl \
    --clip_feature_save_path ./video_features/official_omnivore \
    --output_save_path ./video_features
    
 python prepare_ego4d_dataset.py \
    --input_train_split ~/ego4d_data/v1/annotations/nlq_train.json \
    --input_val_split ~/ego4d_data/v1/annotations/nlq_val.json \
    --input_test_split ~/ego4d_data/v1/annotations/nlq_test_unannotated.json \
    --video_feature_read_path ~/ego4d_data/v1/slowfast8x8_r101_k400 \
    --clip_feature_save_path ./video_features/official_slowfast \
    --output_save_path ./video_features
```
* Run the following command to concatenate all three features together
```shell
python fuse.py
```
* After that, you can use video features under video_features/fusion to run the experiment (you can delete the other folders under video_features or keep them to reproduce our results with slowfast/omnivore/egovlp features).