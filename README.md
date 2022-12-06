# Hand-Gesture-Recognition---Temporal Convolution 1D and LSTM

This repository is a demo tensorflow implementation of the deep learning model for hand gesture recognition introduced in the article [Deep Learning for Hand Gesture Recognition on Skeletal Data](https://ieeexplore.ieee.org/document/8373818) from G. Devineau, F. Moutarde, W. Xi and J. Yang.

I am going to implement RNN with LSTM layers for this problem. This will be my first experience with RNN.

## Getting started

The notebook below, provided by google collab, includes gesture data loading, model creation, and model training.

![Overview of the gesture recognition approach with a CNN deep leaning model](images/pipeline.png)

Additional details are provided below.

---

## Model overview

##### Summary

The neural network extracts motion features, using a dedicated temporal feature extractor (right) made of temporal convolutions. These temporal features are finally used to determine the nature of the gesture performed.

##### Model input

Each hand joint typically has 2 or 3 dimensions, to represent its (x,y) or (x,y,z) position in space at a given timestep. A gesture is thus represented by a sequence over time of n_joints (e.g. 22 joints in the image above) joints, or, equivalently by a sequence over time of n_channels (e.g. 66 channels = 22 joints x 3 channels: for x, y and z position of the joint).

The model use such sequences as input.

#########Data format

The model expects gestures to be tensors of the following shape: ```(batch_size, duration, n_channels)```.

##### Finding temporal features

The key idea of the model is to extract relevant features to classify the gesture, based on the temporal evolution of each channel/signal. (The extracted features will later be used to perform the classification.)

Each channel is processed separately.

##### Gesture classification

Once features have been extracted for each channel, they need to be “merged”.

To that extent, they are all fed into a dense neural network (one hidden layer) which performs the final classification.

The full model (by-channel temporal feature extraction + final MLP) is differentiable and can be trained end-to-end.

---

### Training the model with your own gestures

Recognizing hand gestures can be useful in many daily real-life situations: writing, drawing, typing, communicating with sign language, cooking, gardening, driving, playing music, playing sport, painting, acting, doing precise surgery, pointing, interacting with one’s environment in augmented reality or virtual reality, for drone control, lights control, sound control, home automation, medicine, nonverbal communication, … the list is almost limitless!

##### A. Get hand pose sequences

If you only have videos of the gestures, first use a vision-based approach to generate hand pose/skeletal representation of your sequences.

For instance, you can use the CMU's OpenPose demo:  https://github.com/CMU-Perceptual-Computing-Lab/openpose or Google's MediaPipe https://github.com/google/mediapipe/blob/master/mediapipe/docs/hand_tracking_mobile_gpu.md

You can also use motion capture gloves, or dedicated sensors (Leap Motion, RealSense camera, ...) that directly provide such sequences.

If you don't have any hand gesture available or if you want to reproduce the results of the research article, you can also download the [DHG-14/28 Dataset](http://www-rech.telecom-lille.fr/DHGdataset/) or the [SHREC’17 Track Dataset](http://www-rech.telecom-lille.fr/shrec2017-hand/). To preprocess the two datasets, and load them with a single line of code, [please follow the instructions provided here](https://github.com/guillaumephd/deep_learning_hand_gesture_recognition/issues/1).

##### B. Data format

The model expects gestures to be tensors of the following shape: ```(batch_size, duration, n_channels)```.

For instance, if you want to recognize a batch of 32 gestures of length 100 on a hand skeletal constituted of 22 joints in 3D (x,y,z), i.e. 22x3=66 channels, the shape of the tensor should be : ```(32, 100, 66)```.

##### C. Load the gesture sequences data

First change the ```load_data()``` and the functions just below it to load your own data (section 2 of the notebook).

Then load it (section 4 of the notebook).

##### D. Train the model

Specify how many gesture classes you have and how many hand channels (not joints) you have (section 4 of the notebook).

You can now train the model on your own model.

##### E. Training tips

1. You should always visualize the metrics (loss, accuracy, ...; both for train/test/validation) of the model during the training. You can use ```tensorboard``` for that (see notebook for more details).

1. If the validation error starts increasing you've likely already been overfitting for a few epochs
2. If you encounter some ```pytorch``` error, ensure that your labels are in the correct range of values, and that the gesture tensor shape is correct
3. If the accuracy curves are too noisy or not smooth enough, consider decreasing the optimizer's learning rate
4. You can use data augmentation to improve the model performance

---

### Requirements

The notebook will run fine with:

- `python 3`

- `pytorch 1.0` and above

- Usual pip modules: `numpy` `sklearn` `scipy`

Optional: if you want to monitor the neural network’s metrics during training, you’ll also need ```tensorboardX``` and ```tensorboard``` (the latter is provided by ```tensorflow```)

### Citation

If you find this code useful in your research, please consider citing:

```
@inproceedings{devineau2018deep,
  title={Deep learning for hand gesture recognition on skeletal data},
  author={Devineau, Guillaume and Moutarde, Fabien and Xi, Wang and Yang, Jie},
  booktitle={2018 13th IEEE International Conference on Automatic Face \& Gesture Recognition (FG 2018)},
  pages={106--113},
  year={2018},
  organization={IEEE}
}
```
