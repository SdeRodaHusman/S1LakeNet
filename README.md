# **S1LakeNet**  

## **Author**  
**Sophie de Roda Husman**  
PhD Candidate, Geoscience and Remote Sensing  
Delft University of Technology  

📧 **Email:** [S.deRodaHusman@tudelft.nl](mailto:S.deRodaHusman@tudelft.nl) / [sophiederoda@hotmail.com](mailto:sophiederoda@hotmail.com)  
🐦 **Twitter:** [@SdeRodaHusman](https://twitter.com/SdeRodaHusman)  
🐱 **TikTok:** [@zuidpool_sophie](https://www.tiktok.com/@zuidpool_sophie)  

## **Overview**  
This repository contains the code and materials used for the final chapter of my dissertation:  
📄 **[From Pixels to Puddles: Mapping Surface Melt on Antarctic Ice Shelves](https://research.tudelft.nl/en/publications/from-pixels-to-puddles-mapping-surface-melt-on-antarctic-ice-shel)**  

### **What is S1LakeNet?**  
S1LakeNet is a deep learning model designed to classify Antarctic supraglacial lakes at the end of the melt season. Specifically, it predicts whether a lake is **refreezing** or **draining**.  

- **Input:** Sentinel-1 time series  
- **Model:** ConvLSTM (Convolutional Long Short-Term Memory)  
- **Training Data:** Supraglacial lakes in Greenland  
- **Prediction Output:**  
  - **Probability < 50% →** Lake is refreezing  
  - **Probability > 50% →** Lake is draining  
