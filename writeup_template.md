[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/hog_feature.png
[image3]: ./examples/multislide_windows.png
[image4]: ./examples/test5_windows.png
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
[image5]: ./examples/test5_heatmap.jpg
[image6]: ./examples/no_heatmap_cut.png
[image7]: ./examples/heatmap_cut1.png
[image8]: ./examples/result_YCrCb.png
[video1]: ./project_video.mp4
[video2]: ./test_video.mp4

---
## Histogram of Oriented Gradients (HOG) and Car Classifier


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


Also, I used LUV and hog_channel=0 and created successful result with the same code and slightly different cuts for averaged heatmaps for consecutive frames.

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

I think we can get the same results for other configurations with slightly different cuts in averaged heatmaps for the consectuve frames.

#### Classification
After extracting features, we used StandarScaler from sklearn. We divided samples to traing sets and test sets with ratio of 0.8:0.2. Also, we cross-checked
by traing GTI data and testing on Extra and KITII data. They gave consistent accuracies. 
We used Linear SVC, Naive Bayes, Random Forest, Gradient Boosting and other non-linear SVM models. They all gave close to 99% accuracy. 

So, i chose linearSVC because prediction time during video processing is significantly less than for other models. 
The model files for two configurations we used in this study can be found in home dir:

        svc_params_LUV_hog0_orient9_size32_pix8_cell2.pk
        svc_params_YCrCb_hogALL_orient9_size32_pix8_cell2.pk


### Sliding Window Search

Codes for the following studies can be found in VehicleDetection.ipynb. Functions can be found in lesson_functions.py

I used multi-scale windows. The following scales have overlap of 75% and are chosen empircilly. 
	
	scale1  : ystart = 380   ystop=480   scale=1    
	scale2  : ystart = 400   ystop=600   scale=1.5
	scale3  : ystart = 500   ystop=700   scale=1.5

The sliding windows for the LUV configuration are shown below.

![][image3]

The sliding windows and headmap for image test5.jpg. The image does not have signal for the third scale sliding windows :

![][image5]

### Image processing

All test images without heatmap cut. The rectangles for 3 scale sliding windows are combined and calculated heatmap. 
We can see that all cars are detected with lot of false signals.

![][image6]

#### The image result for LUV configuration

With heatmap cut, we cleaned false signals and removed some weak true signals(white cars) 

![][image7]

#### The image result for YCrCb configuration

we got much better results for the test YCrCb images. Depending on the road condition and type of colors of the cars in images, 
some color spaces and hog configurations show better results than others in certain cases. It does not mean those are better than others.

![][image8]


---

### Video Processing


We created class which keeps all rectangles and heatmaps for thirty consecutive frames. We did not use heatmap cuts when we save heatmap.
The reason we are not using heatmap cuts is we want to keep as many signal as possible since we can decrease false signals comparing and 
matching consecutive frame information.

The heatmap values are converted to zero or one and saved.

	heat = add_heat(heat,rectangles)
	heat[heat>0]=1
	det.add_heat(heat)
	det.add_rects(rectangles)
 
Using the normalized heat, we tried to separate true signals from false signals :
    	
	heatmap_cut = np.zeros_like(img[:,:,0])
	for heat in det.prev_heat:
        	heatmap_cut = heatmap_cut+heat
	
	if color_space=='LUV':
                heatmap_cut = apply_threshold(heatmap_cut, 27)
    	elif color_space=='YCrCb':
                heatmap_cut = apply_threshold(heatmap_cut, 15)      (Cut 1)

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
  
    	if color_space=='LUV':
                heatmap_img = apply_threshold(heatmap_img, i)
    	elif color_space=='YCrCb':
                heatmap_img = apply_threshold(heatmap_img, i//2)    (Cut 2)
     
We have to tune two cuts(Cut 1 and 2). We tuned cut 2 to include all true signals and then, tune cut 1 to exclude false signals as much as possible.

Overall, the procedure is quite robust and we created successfull results using two HOGs configurations without much work.
You can find above code in VehicleDetection.ipynb. The final code with lane detection is implemented in LineAndVehicleDetection.ipynb
The image calibration file is copied to output_images. The functons related to lane finding are written advanced_line.py.

The final videos with lane detection for the LUV configuration:

1. [project video with lane detection for LUV configuration ](./project_video_out_luv32_withlane.mp4)
2. [test video with lane detection for LUV configuration](./test_video_out_luv32_withlane.mp4)

Videos for YCrCb configuration :

1. [project video with lane detection for YCrCb configuration ](./project_video_out_YCrCb32_withlane.mp4)
2. [test video with lane detection for YCrCb configuration](./test_video_out_YCrCb32_withlane.mp4)

---

### Discussion

The procedure i used works well. I used two HOG configurations(two color spaces) and produced successful videos for both. 
Because riding conditions constantly change in videos, its better to use approach independent of color spaces and other parameters.
Second, matching heatmaps and rectangles between consecutive frames has much more upside. I used simple averaging method matched with 
some kind of normalized heatmaps. We can use some pattern matching and simple machine learning models between consecutive 
frames and create method which works better in different road conditions.




