\# Landmark Recognition System



\## Overview

This project implements a landmark recognition system using deep learning techniques. It identifies famous landmarks from images.



\## Dataset

Images were collected using web scraping (Bing Image Crawler).  

Landmarks used:

\- Taj Mahal

\- Eiffel Tower

\- India Gate

\- Big Ben



\## Methodology

\- Transfer Learning using MobileNetV2

\- Image preprocessing (resize, normalization)

\- Classification using Softmax layer



\## Model Details

\- Input size: 128x128

\- Pretrained model: MobileNetV2

\- Training epochs: 5



\## Results

\- Accuracy achieved: \~96.7%



\## How to Run

1\. Install dependencies

2\. Run train\_mobilenet.py

3\. Run predict.py



\## Future Improvements

\- Use larger dataset (Google Landmarks Dataset)

\- Fine-tune pretrained model

\- Deploy as web application

