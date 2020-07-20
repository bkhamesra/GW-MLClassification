# Classifying Gravitational Wave Source using Supervised Machine Learning -

## Authors - 
1. Bhavesh Khamesra (Project Lead)
2. Kamal Sharma
3. Ravindranadh Bolla

## Goal - 
Gravitational waves are ripples in space and time created by the motion of massive accelerated object(s). Their recent detection in 2015 led to complete new way to explore the universe, especially the regions which do not emit strong electromagnetic signatures. One of the strongest sources of gravitational waves are the black hole and neutron stars binary systems. Due to their huge mass and compact size, these objects induce a strong curvature on fabric of space and time.  As these objects rotate around each other, the curvature shifts creating ripples much like the ripples generate in water due to motion of two boats in a circle. These ripples contains the information of their sources and can travel for millions of light years without loosing their original form. Today, using gravitational wave detectors, we can learn more about the universe and locate several hidden objects such as black holes. 

In this work, we look at the gravitational wave signatures emitted by binary black holes using Supervised Machine Learning methods. A binary black hole system can be described in terms the 10 intrinsic parameters - two component masses, two spin vectors (total 6 spin components), initial orbital frequency and initial eccentricity. One of the primary ways to distinguish these systems is based on their spins - Non Spinning where both black holes are non-rotating, Aligned-Spin, where the initial spins of each black holle are only in z direction, perpendicular to the orbital plane, and Precessing where spins can be in any direction. We would like to apply Supervised Machine Learning tools to see if such classification can be achieved using machine learning methods. 

## Methodology - 

* Data Ceaning and Extraction - Python 
* Feature Extraction and Feature Selection - Python (sklearn library), Weka
* Supervised ML - Python (sklearn), Weka
* Unsupervised Learning - Weka



## Directory And File Structure - 

1. Data - This directory contains the dataset used for this work. 
   * WaveformData - Text data obtained from original NR waveforms h5 files (obtained from publicly available dataset from each group's websites). Here we only extract the most dominant (2,2) mode. 
   * Metadata     - Contains information about each waveform and initial binary system such as  mass ratio, spins, eccentricity etc
   * FilteredData - After cleaning the waveform data, the cleaned data is stored in this directory. 
2. Scripts - Check the Jupyter notebooks for detailed analysis and results
   * Bhavesh - Contains Jupyter nobooks and other python scripts for data cleaning  
3. Results - 
   * WEKA_Results - This includes Results obtained using WEKA
   * Poster - A ppt file of poster presented at Georgia Regional Astronomy Meeting, 2017.
    
