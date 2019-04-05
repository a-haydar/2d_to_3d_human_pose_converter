# 2D to 3D Human Pose converter for generating Social Robot Pose
This code base accompanies the repository for generating Co-Speech Gestures in Social Robots, that can be found [here](https://github.com/pieterwolfert/co-speech-humanoids).
The code in this repository is forked from [here](https://github.com/youngwoo-yoon/2d_to_3d_human_pose_converter),
and adapted to better learn 3D representations of 2D frontal poses, including head pose.

----
### Installation
The code is implemented in Python 3.7 and PyTorch 1.0

You need three folders: data, models and panoptic_dataset.

From the CMU Panoptic [Dataset](http://domedb.perception.cs.cmu.edu/dataset.html) you download one or more samples, and place the folders with the name 'hdPose3d' in the folder 'panoptic_dataset'.

Second, you use run 'generate_dataset.py' to create a pickled dataset, that can be used for training and test.
Third, run train.py, make sure to adjust the hyperparameters to your need.
At last, you can visualize the trained results using 'test.py'

Training can easily be done on CPU.

### Pre-trained models
TBA. Current hyperparameters result in a validation loss of 0.027.

### Credits
Many thanks to Youngwoo Yoon for providing his code.
