# CU2O_segmentation
This repository contains codes used to perform atomic-level segmentation of high resolution TEM images containing heterogeneous backgrounds. 
main.ipynb contains codes to train the segmentation model or to use the pretrained model to perform inference. Analysis code include scripts to obtain centroids of atoms from the model predictions and to obtain the true positive, false positive, and false negative.
The data_preparation folder contains notebooks to create manual labels of the particles given a image (HRTEM.tiff) and its inverse fast Fourier transform (IFFT.tiff).