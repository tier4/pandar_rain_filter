# Pandar Rain Filter

Command-line Interface for Pandar rain filtering tool. The tool expects a bag file containing pointclouds from `/pandar_points_ex` topic.

## Usage

```sh
rosrun pandar_rain_filter rosbag_processor_pandar_rain_filter _file_path:=[bag file path] _output_path:=[output_path] _no_rain_pcd_path:=[no_rain_pcd] _train_val_selection:=[train/val]
```

# Parameters and options

 |Param|Default Value|Description|
 |---|---|---|
 |`file_path`|None|Path of bag file to be processed|
 |`output_path`|None|Path where the range images and labels will be stored|
 |`no_rain_pcd_path`|None|Path of the no rain point cloud pcd|
 |`train_val_selection`|None|Training or validation data|


The data will be stored in `output_path` with the following structure:

```
└── output_path
    ├─particle_labels
    │   ├─train
    │  	│ ├─0.png
    │   │ ├─1.png
    │   │ ├─ ...
    │   └─val
    │  	  ├─0.png
    │     ├─1.png
    │     ├─ ...
    └─range_images
        ├─train
        │ ├─first_depth
        │ │ ├─0.png
        │ │ ├─1.png
        │ │ ├─ ...
        │ ├─first_intensity
        │ │ ├─0.png
        │ │ ├─1.png
        │ │ ├─ ...
        │ ├─first_return_type
        │ │ ├─0.png
        │ │ ├─1.png
        │ │ ├─ ...
        │ ├─last_depth
        │ │ ├─0.png
        │ │ ├─1.png
        │ │ ├─ ...
        │ ├─last_intensity
        │ │ ├─0.png
        │ │ ├─1.png
        │ │ ├─ ...
        │ └─last_return_type
        │   ├─0.png
        │   ├─1.png
        │   ├─ ...
        └─val
          ├─first_depth
          │ ├─0.png
          │ ├─1.png
          │ ├─ ...
          ├─first_intensity
          │ ├─0.png
          │ ├─1.png
          │ ├─ ...
          ├─first_return_type
          │ ├─0.png
          │ ├─1.png
          │ ├─ ...
          ├─last_depth
          │ ├─0.png
          │ ├─1.png
          │ ├─ ...
          ├─last_intensity
          │ ├─0.png
          │ ├─1.png
          │ ├─ ...
          └─last_return_type
            ├─0.png
            ├─1.png
            ├─ ...
