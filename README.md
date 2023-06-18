# TRAFFIC LIGHT DETECTION

## Steps

1. Use pretrained dataset, here `ssd_resnet50_v1_fpn_640x640_coco17_tpu-8`, to extract traffic light images from any of the datasets.
2. Prepare the dataset for training by manually separating the images into 4 folders: `0_green, 1_yellow, 2_red, 3_not`.
3. Use the separated images to train a new model, here image.

## Code/Directory Structure

* [dataset](./dataset)
  * [archive](./dataset/archive)
    * [test_dataset](./dataset/archive/test_dataset)
      * [test_images](./dataset/archive/test_dataset/test_images)
    * [train_dataset](./dataset/archive/train_dataset)
    * [train](./dataset/archive/train_dataset/train)
    * [train_1to500](./dataset/archive/train_dataset/train_1to500)
    * [train_images](./dataset/archive/train_dataset/train_images)
  * [detected_and_cropped](./dataset/detected_and_cropped)
    * [archive_cropped](./dataset/detected_and_cropped/archive_cropped)
    * [S2TLD(720x128-cropped](./dataset/detected_and_cropped/S2TLD(720x128-cropped))
  * [S2TLD(720x128)](./dataset/S2TLD(720x128))
    * [normal_1](./dataset/S2TLD(720x128)/normal_1)
      * [Annotations](./dataset/S2TLD(720x128)/normal_1/Annotations)
      * [JPEGImages](./dataset/S2TLD(720x128)/normal_1/JPEGImages)
    * [normal_2](./dataset/S2TLD(720x128)/normal_2)
    * [Annotations](./dataset/S2TLD(720x128)/normal_2/Annotations)
    * [JPEGImages](./dataset/S2TLD(720x128)/normal_2/JPEGImages)
  * [train_traffic_light](./dataset/train_traffic_light)
    * [0_green](./dataset/train_traffic_light/0_green)
    * [1_yellow](./dataset/train_traffic_light/1_yellow)
    * [2_red](./dataset/train_traffic_light/2_red)
    * [3_not](./dataset/train_traffic_light/3_not)
  * [zips](./dataset/zips)
* [__pycache__](./__pycache__)
* [src](./src)
* [test](./test)
  * [small_v1](./test/small_v1)
  * [small_v2](./test/small_v2)
  * [test_images](./test/test_images)
  * [test_set_output](./test/test_set_output)
  * [video](./test/video)
* [train](./train)
  * [traffic_light_dataset](./train/traffic_light_dataset)
  * [0_green](./train/traffic_light_dataset/0_green)
  * [1_yellow](./train/traffic_light_dataset/1_yellow)
  * [2_red](./train/traffic_light_dataset/2_red)
  * [3_not](./train/traffic_light_dataset/3_not)
* [trained_models](./trained_models)
