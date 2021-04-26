# Pandar Rain Filter

Command-line Interface for Pandar rain filtering tool. The tool expects a bag file containing pointclouds from `/pandar_points_ex` topic.

## Usage

```sh
rosrun pandar_rain_filter rosbag_processor_pandar_rain_filter _file_path:=[bag file path] _output_path:=[output_path] _no_rain_pcd_path:=[no_rain_pcd]
```

# Parameters and options

 |Param|Default Value|Description|
 |---|---|---|
 |`file_path`|None|Path of bag file to be processed|
 |`output_path`|None|Path where the range images and labels will be stored|
 |`no_rain_pcd_path`|None|Path of the no rain point cloud pcd|


The data will be stored in `output_path` with the following structure:

```
├── output_path
│    ├── labels (png images)
│    └── range_images (EXR files)
