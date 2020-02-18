# WIldLife Protection Project

This project aims to provide a visual survelliance system for a Wildlife reservation park.
The objectives this system achieve are:

1. Detect animal captured by trap camera. This can help park administrators to track animals especially endangered animals.
2. Runs a survelliance camera to monitor activities in the reserved park, especially banned activities.

## Trap Camera Animal Detection

The trap camera mostly found in the parks are triggered when a moving object pass in front of the camera. The cameras mostly employs motion sensors.
In this case, the picture captured by the camera is processed by some on-site computer which runs the animal classification model and returns the detected animal.

### Animal Classification Model

 Deep Learning Architecture - Inceptionv3 transfer learning. Check retrain.py file for how the transfer learning is achieved and check out the Inceptionv3 architecture itself to learn more about the architecture.
 Dataset - The dataset contains 27 classes of animals mostly collected from Google Image search. The classes are: Bear, Butterfly, California bighorn sheep, Cat, Cow, Deer, Dog, Elephant, Fox, Gorilla, Hen, Horse, Jaguar, Leopard, Lion, Monkey, Orangutan, Pangolin, Panther, Rhinoceros, Saola, Sheep, Spider, Squirrel, Tiger, Wolf and Zebra.
 Training method - retrain.py is well documented and fully describes the training method. For a breakdown, you can see my blog post [How to classify pothole images using transfer learning](https://medium.com/@hksalaudeen/tutorial-how-to-classify-pothole-images-using-transfer-learning-c6782e8ff036)
 Classification method - label_image.py gives detailed explanation on how the classification works.

### Object Detection Model
The survelliance camera runs on this model. It detects objects in video stream and downloads the same video but with detected object.
The model can detects about 1000 classes of objects based on ImageNet dataset.
Deep Learning Architecture - YOLOv3. The object detector steps can be found here: [YOLO Object Detection with OpenCV](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)
Dataset - ImageNet 1000 objects
Video Processing - OpenCV is used to read video file and process frame by frame. Python-ffmpeg is used to write the processed frames to file and save to the computer.

