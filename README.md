# JPNTAXI Rosbag Extractor

Command-line Interface for Jpntaxi Rosbag processor.

## Usage

```sh
./rosbag_process.sh -d dataset_base_dir -b bag_name -t taxi_id -f frame_rate -s skip_first -p s3_path
```

# Parameters and options

 |Param|Default Value|Description|
 |---|---|---|
 |`dataset_base_dir`|None|Base directory containing bag files. The bag files need to be inside a folder named `raw`. Same directory is used for output of results|
 |`bag_name`|None|Name of bag file to be processed|
 |`taxi_id`|0|Id of the taxi|
 |`frame_rate`|10|Output frame rate|
 |`skip_first`|0|Number of frames to skip at the start|
 |`s3_path`|None|Base s3 path to insert in json files|


The data will be stored in `dataset_base_dir` with the following structure:

```
├── dataset_base_dir
│   └──extracted_frames
│      ├── images (png images)
│      └── json (json encoded data)
```