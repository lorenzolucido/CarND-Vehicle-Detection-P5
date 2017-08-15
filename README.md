

# **Vehicle Detection Project**
#### _Lorenzo's version_
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car_not_car]: ./car_not_car.png
[hog_car]: ./hog_car.png
[hog_non_car]: ./hog_non_car.png
[sliding]: ./sliding-window.png
[heating]: ./pipeline_heating.png
[labeling]: ./pipeline_labeling.png


### [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

---
_Important Note: I have put all the code in a single Jupyter notebook called [`P5-Vehicle-Detection.ipynb`](./P5-Vehicle-Detection.ipynb).
And here is the [HTML](./P5-Vehicle-Detection.html) version._

### 1. Training a classifier

#### Histogram of Oriented Gradients (HOG)

The code for this step is contained in the section 2a of the Jupyter notebook.
I implemented the `HOGFeatExtractor` as a [sklearn pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) `transformer`, so that I can leverage on the pipeline ability to chain various transformations (vertically and horizontally) and easiness to change parameters.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][car_not_car]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HLS` color space (L-Channel) and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` (2 examples for each class we are trying to predict):

![alt text][hog_car]
![alt text][hog_non_car]

#### Histogram of colors (`histCol`) and Spatial Binning (`spaBin`) features (sections 2b and 2c)

As a way to further help the classifier, I also extracted [histogram of color](https://en.wikipedia.org/wiki/Color_histogram) and spatial binning features. Both have been implemented as sklearn pipeline `transformers`.

#### Standardization and Classification (_&rarr;Notebook section 3._)

I then  standardized the 3 sets of features by passing them through a sklearn `StandardScaler`, and finally trained a **Linear SVM classifier** with all these features. The name of the full classification pipeline is `p_classify` in the notebook.
In order to set the various parameters, I looked at my test set accuracy and tried to maximize it.
I eventually obtained an accuracy of **99.0%** on the test set.

_Note: The pipeline module could have been leveraged to run cross-validation on a parameter grid._

### 2. Vehicle detection on images and videos

#### Sliding Window Search (_&rarr; Notebook section 4._)

So I had a working classifier for 64x64 images. I then implemented a sliding window technique in order to spot where are the cars on 1280x720 images. In other words, I extracted multiple sub-images from the big (1280x720) image and run the classifier on them in order to predict if there is any car on that particular sub-image.

Here is an example output for one of the test images with sliding window 128x128:

![alt text][sliding]

#### Optimizing the Sliding Window Search (_&rarr; Notebook section 5._)

In order to improve the performance of the `SlidingWindowClassifier`, I had to modify a few items in my pipeline:
1. take only the bottom part of the big 1280x720 image (since basically the top part is
2. cache the Histogram of Gradients for remaining part of the big image and then re-use subset of those features for sub-images (see `SlidingWindowHOG` class in section 4.)
 just the sky!)
3. run the `SlidingWindowClassifier` with multiple window sizes and make sure they do not overlap too much, as this would classifying a lot of images
4. set the classifier probability threshold for keeping windows with cars much higher than 50% in order to avoid false positives (eventually set at 90%)
5. aggregate the output of the multiple `SlidingWindowClassifier` into a heatmap which sums the probabilities of multiple windows with a minimum threshold
6. label the heated areas

Below are samples taken after the step 5 (_heating_):
![alt text][heating]

Below are samples taken after the step 6 (_labeling_):
![alt text][labeling]

---

### Video Implementation

Here's a [link to my video result](./project_video_output.mp4)


---

### Discussion

Clearly, filtering out false positive was a big challenge in this project. I would think there are chances that the current pipeline would not perform well if the weather conditions / light are different for example. A neural network with enhanced data would probably be a reasonable choice as I can see it being more robust to these problems.
Also, it took roughly 30 minutes to compute a 50 seconds video, HOG caching was a good improvement (from 5 seconds per frame to 1.5 second per frame) but this is still not good for real-time. A C++ implementation might be required here for real-time car tracking.
