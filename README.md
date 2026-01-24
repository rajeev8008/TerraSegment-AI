# TerraSegment-AI

**Real-time AI-powered semantic segmentation for aerial and satellite imagery.**


---

##  Project Overview
Real-time AI-powered semantic segmentation of aerial imagery. Upload satellite/drone images to instantly classify terrain into 6 categories (water, roads, vegetation, buildings, land) using a trained U-Net deep learning model. Includes an interactive database search feature to find similar images by terrain composition.This system is designed to provide geospatial insights for applications such as urban planning, environmental monitoring, agricultural land-use analysis, and disaster response assessment(maybe).

###  Key Features
* **Real-Time Classification**: Instant pixel-wise segmentation into Water, Roads, Vegetation, Buildings, Land, and Unlabeled classes.
* **Composition-Based Search**: A unique database feature allowing users to filter and find images by specific terrain percentages using interactive sliders.
* **High-Resolution Support**: Optimized handling for high-resolution drone and satellite data.

---

## Interface & Visualization

###  Semantic Segmentation
> **The core of TerraSegment-AI uses a U-Net architecture, a specialized convolutional neural network designed for fast and precise pixel-wise classification. It processes high-resolution aerial imagery by breaking it into patches to maintain computational efficiency while capturing both fine-grained structural details and broader environmental context.**

<img width="1896" height="939" alt="image" src="https://github.com/user-attachments/assets/3a180c0d-3345-4a95-9d6d-a9404aa4c182" />

---

###  Terrain Composition Search
> **Adjust the sliders to define a desired terrain composition and instantly retrieve matching satellite images from the database. This feature leverages a metadata indexing system that calculates the percentage of each terrain class for every segmented image, allowing users to query for specific environmental profiles.**

<img width="1881" height="913" alt="image" src="https://github.com/user-attachments/assets/2a1c4b18-1864-415f-b00c-2b28123b0f1d" />
<img width="1890" height="899" alt="image" src="https://github.com/user-attachments/assets/8f1d301e-ab5e-40da-849b-0b71ee1745f1" />

---

**(work in progress)**

