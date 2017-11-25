# U-Net-Segmentation
3D U-Net Implementation Based on TensorFlow

To Do List:
1. conv_def: Dilated Convolution 
2. loss_def: Problem with dice loss, softmax to be checked, remove dice~
3. data_io: other data argumentation, normalization over the whole dataset
4. new: improve the sampling method -> cube or go all
5. *test function!
6. *rewrite output format, separate train and test
7. paper!
8. model: feature number variant

1. dice: without softmax
2. argumentation: rotation and flipping
3. regularization: AdamOptimizer
4. loss weight: zero loss problem? pure label in?*
5. **data_analysis code -> plot
6. **dilated convolution
7. cut out unnecessary ones of testing?*
