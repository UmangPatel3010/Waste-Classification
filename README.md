# Waste Classification using CNN

This project implements a Convolutional Neural Network (CNN) to classify waste images into two categories: Organic and Recyclable. The goal is to automate waste segregation by identifying the type of waste from an image, contributing to efficient recycling and waste management.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This waste classification model uses deep learning techniques to classify images of waste. The CNN architecture enables the model to learn important features from images, distinguishing between organic and recyclable waste. The model can be integrated into smart waste management systems to facilitate efficient sorting.

## Dataset
The [dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data) consists of labeled images of organic and recyclable waste. The images are preprocessed and augmented to improve model performance.


### Dataset Structure:
- **Organic:** Food waste, garden waste, etc.
- **Recyclable:** Plastics, paper, glass, metals, etc.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/waste-classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd waste-classification
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run Project:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Prepare your dataset by placing images into the respective folders (Organic and Recyclable).
2. Train the model:
   ```bash
   python ModelCreation.py
   ```

## Results
The model achieves high accuracy in distinguishing between organic and recyclable waste.

## Contributing
Contributions are welcome! If you would like to improve the model, add new features, or fix bugs, feel free to submit a pull request.

