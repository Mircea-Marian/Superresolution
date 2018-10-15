# Superresolution

The application transforms 255x255 image into a 510x510 one through the use of 3 separate CNNs. 
The training set is obtained by splitting a 510x510 picture into 4  255x255 images: P1, P2, P3, P4.
P1 is used as input for the CNNs, while the rest represent the output. Due to limited hardware resources, the results are not that well defined, but the CNNs seem to learn to determine contours and colors.
