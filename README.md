# AlexNet-for-joint-fluid-identification-and-classification
We use a modified AlexNet to perform 6-class classification on bone joint fluid


#Model
This system is implemented based on the AlexNet deep convolutional model. AlexNet was proposed in 2012 and won the championship of the ImageNet competition that year. In recent years, more and more advanced deep convolutional models have been proposed, such as ResNet101, Inception v3, GoogLeNet, etc. However, their model depths are getting deeper and deeper, requiring more and more samples during training. Considering the sample size and functional requirements of this system, we adopt the most basic AlexNet model (Figure 1) for implementation.
Based on the actual situation of this system, we have made modifications to the AlexNet model, and the modified model is shown in Figure 2. The specific modifications are as follows:
1) Change the dimensions of the first and second fully connected layers to 512;
2) Change the dimension of the last fully connected layer to 6;


#Image Preprocess
The original test strip image samples vary in size, so the system first adjusts all test strip images to a uniform size of 100*100 using linear interpolation.
1) Use random cropping to crop 100*100 images into 80*80 images. Random cropping increases the diversity of samples to some extent, which helps improve model performance.
2) Scale each pixel value on each channel of the image from 0-255 to 0-1.
3) Shift each pixel value to the range of (-1,1).

#classification 
Based on the number of white blood cells, the test strips are categorized into the following six classes: test strips with a white blood cell count in the range of [0,1000) belong to Class 1; test strips with a white blood cell count in the range of [1000,2000) belong to Class 2; test strips with a white blood cell count in the range of [2000,3000) belong to Class 3; test strips with a white blood cell count in the range of [3000,4000) belong to Class 4; test strips with a white blood cell count in the range of [4000,5000) belong to Class 5; and test strips with a white blood cell count in the range of [5000,+∞) belong to Class 6.


