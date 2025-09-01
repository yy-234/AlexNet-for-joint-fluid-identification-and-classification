# AlexNet-for-joint-fluid-identification-and-classification
We use a modified AlexNet to perform 6-class classification on bone joint fluid

#Image Preprocess

The original test strip image samples vary in size, so the system first adjusts all test strip images to a uniform size of 100*100 using linear interpolation.
1) Use random cropping to crop 100*100 images into 80*80 images. Random cropping increases the diversity of samples to some extent, which helps improve model performance.
2) Scale each pixel value on each channel of the image from 0-255 to 0-1.
3) Shift each pixel value to the range of (-1,1).

#classification 

Based on the number of white blood cells, the test strips are categorized into the following six classes: test strips with a white blood cell count in the range of [0,1000) belong to Class 1; test strips with a white blood cell count in the range of [1000,2000) belong to Class 2; test strips with a white blood cell count in the range of [2000,3000) belong to Class 3; test strips with a white blood cell count in the range of [3000,4000) belong to Class 4; test strips with a white blood cell count in the range of [4000,5000) belong to Class 5; and test strips with a white blood cell count in the range of [5000,+∞) belong to Class 6.

predict_model.py is the code file for testing data

label.txt is an example of the used label


   [Pre trained models are saved in
](https://huggingface.co/YU123ing/AlexNet_modified/blob/main/final_model.pth)  Click on this link to download the pre trained model.
