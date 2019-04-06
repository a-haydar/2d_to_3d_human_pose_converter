# 2D to 3D Human Pose converter for generating Social Robot Pose
This code base accompanies the repository for generating Co-Speech Gestures in Social Robots, that can be found [here](https://github.com/pieterwolfert/co-speech-humanoids).
The code in this repository is forked from [here](https://github.com/youngwoo-yoon/2d_to_3d_human_pose_converter),
and adapted to better learn 3D representations of 2D frontal poses, including head pose.

----
### Installation
The code is implemented in Python 3.7 and PyTorch 1.0

1. Create these folders in the repository root: data, models and panoptic_dataset.
2. Download one or more samples from the CMU Panoptic [Dataset](http://domedb.perception.cs.cmu.edu/dataset.html).
3. Place the folders from the sample files with the name 'hdPose3d' in the folder 'panoptic_dataset'.
4. Run 'generate_dataset.py' to create a pickle containing the preprocessed dataset.
5. Run 'train.py', hyperparameters can be set in the file.
6. Visualize the results using 'test.py'.

### Pre-trained models
A pre-trained model on 44k samples can be found [here](
https://files.pieterwolfert.com/model_0_0227_.pth). 
Place this model in the folder 'models', and make sure to change the filename in 'test.py'. 

### Credits
Many thanks to Youngwoo Yoon for providing his code.
