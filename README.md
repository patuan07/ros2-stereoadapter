# ros2-stereoadapter
This is a wrapper of StereoAdapter ([check the paper out here](https://github.com/AIGeeksGroup/StereoAdapter)).
## Using this repo:
1. Clone it into your `~/workspaces/your-workspace`
```bash
cd ~/workspaces/your-workspace/
git clone https://github.com/patuan07/ros2-stereoadapter.git
```
2. Download DepthAnythingV2 weights
```bash
cd ~/workspaces/your-workspace/src/stereo_publisher/stereo_publisher/
mkdir -p Depth-Anything-V2/checkpoints
cd Depth-Anything-V2/checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth
```
3. Modify paths: Find all the parts that need to be modified in
```bash
/src/stereo_publisher/launch/stereo_detection.launch.py
/src/stereo_publisher/stereo_publisher/combined_publisher.py
/src/stereo_publisher/stereo_publisher/detection_node.py
/src/stereo_publisher/stereo_publisher/new_depth_estimation_node.py
```
4. Launch with
```bash
ros2 launch stereo_publisher stereo_detection.launch.py
```

