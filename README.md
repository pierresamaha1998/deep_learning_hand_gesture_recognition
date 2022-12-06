# Hand-Gesture-Recognition---Temporal Convolution 1D and LSTM

This repository is a demo tensorflow implementation of the deep learning model for hand gesture recognition introduced in the article [Deep Learning for Hand Gesture Recognition on Skeletal Data](https://ieeexplore.ieee.org/document/8373818) from G. Devineau, F. Moutarde, W. Xi and J. Yang.

#### Summary

A deep learning models are used to classify hand gesture. I am going to implement RNN with ```LSTM``` layers for this problem in addition to the work that have been done in the paper. This will be my first experience with RNN and temporal convolution 1D.

Each hand joint typically has 2 or 3 dimensions, to represent its (x,y) or (x,y,z) position in space at a given timestep. A gesture is thus represented by a sequence over time of n_joints (e.g. 22 joints in the image above) joints, or, equivalently by a sequence over time of n_channels (e.g. 66 channels = 22 joints x 3 channels: for x, y and z position of the joint).

The model use such sequences as input. Data format: The model expects gestures to be tensors of the following shape: ```(batch_size, duration, n_channels)```.

In the previous work, the neural network extracts motion features, using a dedicated temporal feature extractor made of temporal convolutions (1D convolution over time) for each individual 1D channel (e.g. let’s say the channel representing the y position of the wrist). These temporal features are finally used to determine the nature of the gesture performed. Once features have been extracted for each channel, they need to be “merged”. To that extent, they are all fed into a dense neural network (one hidden layer) which performs the final classification. The full model (by-channel temporal feature extraction + final MLP) is differentiable and can be trained end-to-end.

I am going to perform a LSTM model to solve the problem. The goal of RNN models is to extract the temporal correlation between the n-joints by keeping a memory of past position of the joints. The RNN layer is connected to a fully connected layer to get the classification output.

This application is useful if you want to know what kind of gesture is happening in a video. Recognizing hand gestures can be useful in many daily real-life situations: writing, drawing, typing, communicating with sign language, cooking, gardening, driving, playing music, playing sport, painting, acting, doing precise surgery, pointing, interacting with one’s environment in augmented reality or virtual reality, for drone control, lights control, sound control, home automation, medicine, nonverbal communication, … the list is almost limitless!


#### Getting started

The notebook below, provided by google collab, includes gesture data loading, model creation, and model training.

---


#### Citation

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
