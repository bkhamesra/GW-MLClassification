# Classifying Gravitational Wave Source using Supervised Machine Learning -

## Authors - 
1. Bhavesh Khamesra (Project Lead)
2. Kamal Sharma
3. Ravindranadh Bolla

## Goal - 

In this work, we look at the gravitational wave signatures emitted by binary black holes and use Supervised Machine Learning methods to learn more about the source of these waves. A binary black hole system can be described in terms the 10 intrinsic parameters - two component masses, two spin vectors (total 6 spin components), initial orbital frequency and initial eccentricity. These black holes orbit around each other creating pertubation in spacetime which we observe as gravitational waves. A question of interest is then how can we determine the source and its properties from the observed gravitational wave signal.

Here we attempt to use Machine Learning to address part of this question. The aim of this project is two fold - 
1. Identify the spin-type of system 
2. Identify the ratio of masses 



Directory And File Structure - 

1. Data - This directory contains the dataset used for this work. 
   * WaveformData - Text data obtained from original NR waveforms h5 files (obtained from publicly available dataset from each group's websites). Here we only extract the most dominant (2,2) mode. 
   * Metadata     - Contains information about each waveform and initial binary system such as  mass ratio, spins, eccentricity etc
   * FilteredData - After cleaning the waveform data, the cleaned data is stored in this directory. 
2. Scripts - 
   * 
3. Results - 
   * WEKA_Results - This includes Results obtained using WEKA
   * Report - A detailed report explaining the methodolofy and final results. 
   * Poster - A ppt file of poster presented at Georgia Regional Astronomy Meeting, 2017. 
