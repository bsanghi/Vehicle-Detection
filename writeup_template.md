[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/hog_feature.png
[image3]: ./examples/multislide_windows.jpg
[image4]: ./examples/test5_windows.jpg
[image5]: ./examples/test5_heatmap.png
[image6]: ./examples/no_heatmap_cut.png
[image7]: ./examples/heatmap_cut1.png 
[video1]: ./project_video.mp4
[video2]: ./test_video.mp4


# Vehicle Detection 
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and 
create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/hog_feature.png
[image3]: ./examples/multislide_windows.jpg
[image4]: ./examples/test5_windows.jpg
[image5]: ./examples/test5_heatmap.png
[image6]: ./examples/no_heatmap_cut.png
[image7]: ./examples/heatmap_cut1.png
[image8]: ./examples/result_YCrCb.png
[video1]: ./project_video.mp4
[video2]: ./test_video.mp4

---

##Histogram of Oriented Gradients (HOG) and Car Classifier

First, we need to create a classifier which classifies car agains non-cars. Using all the codes we used during the online class, i built pipeline quickly
and tested on the project video. 

We used all data provided for the final model. Also, we did cross-checks,training on the GTI data and testing on Extra-data and KITTI data. They
gave consistent results. The classifier code is giving in the PreprocessingTraining.ipynb and functions are in lesson_functions.py.

### Extracting HOG features
#### Color space and others

I tried several different color spaces. They all gave reasonable results and no one showed significantly better result than others.
There are color spaces that are more suitable for one purpose but bad for the others. 
For the task of classifying cars, I am sure that there is no dedicated color space which works the best.
When one color space is good at detecting white car which located in the grey(sunny) road, its not good for removing false signals.
Its difficult to tell which color space and other HOG features are better when classification results give close numbers.
Second, I looked at other people's github projects. They used YUV,HLS,LUV and YCrCb and created successfull results mostly tuning
parameters.

Since our final goal is to detect cars in video, not in image, I decided to focus on stacked heatmap and rectangles
during video processing. I created class which keeps all heatmap and rectangles for certain number of consecutive frames(30) without any heatmap cuts.
Using simple cuts for averaged heat maps, I removed false signals and detected weak signals(white car in grey background)   

I used mostly default parameters. First, I tried the color space(YCrCb) which used in the online course and created the successful result which detects
cars in the test and project videos all time. 

color_space='YCrCb'
hog_channel='ALL'
orient=9 
pix_per_cell=8 
cell_per_block=2 
spatial_size=(32,32) 
hist_bins=32


After reading the following blog which was mentioned in my mentor's messages, I used LUV and hog_channel=0 and created successful result with the same code 
and slightly different cuts in the slightly different cuts for averaged heatmaps for consecutive frames.

color_space='LUV'
hog_channel=0
orient=9 
pix_per_cell=8
cell_per_block=2
spatial_size=(32,32) 
hist_bins=32
 
The example images for datasets are shown below:

![][image1]

The example hog feature :

![][image2]

For the final result, i chose the second one because its good at detecting white cars on the grey(sunny) background even it gives more false signals on sideways.
You can see the condition in the project video. 
But, we created the successful result(detected cars in the video all the time without false signal) using the first hog configration. I think we can get the same
results for other configurations with slightly different cuts in averaged heatmaps for the consectuve frames. 

#### Classification
After extracting features, we used StandarScaler from sklearn. We divided samples to traing sets and test sets with ratio of 0.8:0.2. Also, we cross-checked
by traing GTI data and testing on Extra and KITII data. They gave consistent accuracies. 
We used Linear SVC, Naive Bayes, Random Forest, Gradient Boosting and other non-linear SVM models. They all gave close to 99% accuracy. 

So, i chose linearSVC because prediction time during video processing is significantly less than for other models. 

### Sliding Window Search

I used multi-scale windows. The following scales have overlap of 75%.  
	scale1  scale2 scale3
ystart = 380    400    500
ystop = 480     600    700
scale = 1       1.5    2.5

The sliding windows are shown below :

![][image3]

The sliding windows for image test5.jpg :

![][image4]

The heatmap for image test5.jpg

![][image5]

All test images without heatmap cut:

![][image6]

All test images with heatmap cut(1):

![][image7]

The reason we are not using heatmap cuts is we want to keep as many signal as possible since we can decrease false signals using consecutive frame information.
Instead of making decision using only one frame info, i prefered to make decision based consecutive frames info. If we are detecting cars only in images, we should
use cuts and focus on color space and hog features. When we used color_space='YCrCb' and hog_channel='ALL', we got much better results.

![][image8]

---

### Video Implementation

The final video :

1. [project_video](./project_video_output.mp4)
2. [test_video](./test_video_output.mp4)

We created class which keeps all rectangles and heatmaps for thirty consecutive frames. The heatmap values are converted to zero or one and saved.

heat = add_heat(heat,rectangles)
heat[heat>0]=1
det.add_heat(heat)
det.add_rects(rectangles)
 
Using the normalized heat, we tried to separate signal from false signal :

heatmap_cut = np.zeros_like(img[:,:,0])
for heat in det.prev_heat:
        heatmap_cut = heatmap_cut+heat

# LUV
cut1=27
heatmap_cut = apply_threshold(heatmap_cut, cut1) # Cut 1
heatmap_cut[heatmap_cut>0]=1  

Then we matched the heatmap passed our cuts with rectangles and only chose rectangles overlapped with heatmap_cut signals. After creating new
rectangle list, we reconstruct 

    heatmap_img = np.zeros_like(img[:,:,0])
    i = 0
    for rect_set in det.prev_rects:
	rect_set_new=check_heat(heatmap_cut,rect_set)
        if len(rect_set_new) > 0:
            i += 1
            heatmap_img = add_heat(heatmap_img, rect_set_new)
    
    # LUV 32
    heatmap_img = apply_threshold(heatmap_img, i) # Cut 2
     
We have to tune two cuts(Cut 1 and 2). We tuned cut 2 to include all true signals and then, tune cut 1 to exclude false signals as much as possible.
Overall, the procedure is quite robust and we created successfull results using two HOGs configurations without much work.

---

###Discussion

The procedure i used works well. I used two HOG configurations(two color spaces) and produced successful videos for both. 
Because riding conditions constantly changes in videos, its better to use approach independent of spaces and other parameters.
Second, matching heatmaps and rectangles between consecutive frames has much more upside. I used simple averaging method matched with 
some kind of normalized heatmaps. We can use some pattern matching or simple linear machine learning models and improve significantly.



