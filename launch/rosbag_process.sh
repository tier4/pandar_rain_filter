#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -v csv_file -d dataset_base_dir -p s3_path -f frame_rate -c frames_count"
   echo -e "\t-csv Csv file containing bag file names and taxi_ids"
   echo -e "\t-d Base directory containing bag files"
   echo -e "\t-p S3 Path"
   echo -e "\t-f Frame rate"
   echo -e "\t-c Max number of frames for final processing"
   exit 1
}

while getopts "v:d:p:f:c:" opt
do
   case "$opt" in
      v ) csv_file="$OPTARG" ;;
      d ) dataset_base_dir="$OPTARG" ;;
      p ) s3_path="$OPTARG" ;;
      f ) frame_rate="$OPTARG" ;;
      c ) frames_count="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$csv_file" ] ||[ -z "$dataset_base_dir" ] || [ -z "$s3_path" ] || [ -z "$frame_rate" ] || [ -z "$frames_count" ]
then
   echo "Some or all of the parameters are missing";
   helpFunction
fi

# Begin script in case all parameters are correct
echo "$csv_file"
echo "$dataset_base_dir"
echo "$s3_path"
echo "$frame_rate"
echo "$frames_count"


source ../../../install/setup.bash

#while loop to process all bagfiles in the csv
OLDIFS=$IFS
IFS=","
count=1 #skip first line of csv
while read bag_name taxi_id <&3; do
   if [ $count -eq 1 ]; then
       count=2;  
       continue
   fi
   zstd -d $dataset_base_dir/raw/$bag_name.bag.zst -f -o $dataset_base_dir/raw/$bag_name.bag
   no_frames=$(rosbag info $dataset_base_dir/raw/$bag_name.bag | grep /sensing/lidar/top/velodyne_packets | grep -oe '\([0-9.]*\)') ## extracts the number of lidar msgs in the bag
   no_frames=$(($no_frames / (10 / $frame_rate))) 
   echo "No. of frames in bag: $no_frames frames"
   if [[ $no_frames -gt $frames_count ]]
   then
        skip_first=`expr $no_frames - $frames_count`
   else
	skip_first=2 ##no frames to be skipped
   fi
   echo "No. of frames to skip: $skip_first frames"


   terminator -T "Preprocessor" -x bash -c "source ../../../install/setup.bash; roslaunch rosbag_preprocessor.launch vehicle_model:=jpntaxi sensor_model:=aip_xx1 vehicle_id:=$taxi_id file_path_bag:=$dataset_base_dir/timestamp_fixed/$bag_name.bag" &

   sleep 10

   rosbag play $dataset_base_dir/raw/$bag_name.bag

#   rosnode kill --all

   pkill terminator

   sleep 5

   if [ -d $dataset_base_dir/extracted_frames/train/folder5/$bag_name ] 
   then
        rm -rf $dataset_base_dir/extracted_frames/train/$bag_name
        echo "Folder already exists: $dataset_base_dir/extracted_frames/train/$bag_name & deleted.. "
   fi

   rosrun rosbag_json_converter rosbag_processor _file:=$dataset_base_dir/timestamp_fixed/$bag_name.bag _output_path:=$dataset_base_dir/extracted_frames/train/$bag_name _s3_path:=$s3_path/$bag_name _frame_rate:=$frame_rate _skip_first:=$skip_first
done 3< $dataset_base_dir/$csv_file
IFS=$OLDIFS
