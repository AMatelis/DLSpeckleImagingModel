# Speckle-Based Flow Rate Estimation

This project applies deep learning to estimate microfluidic flow rates (in µL/min) from grayscale speckle pattern videos using a 3D Convolutional Neural Network (3D CNN). It was developed under the guidance of Dr. Christopher Raub as part of a study on non-invasive optical methods for measuring blood flow.

---

## Background

Speckle patterns arise when coherent light (e.g., laser) scatters through dynamic, heterogeneous media like flowing blood. As the fluid's speed or direction changes, the speckle patterns evolve accordingly. By analyzing sequences of these patterns, we can estimate the underlying flow rate.

---

## Project Overview

### What This Code Does

- Loads short `.avi` speckle videos (named by flow rate, e.g. `5ulpermin.avi`)
- Splits each video into overlapping grayscale frame stacks
- Trains a 3D CNN (`BloodFlowCNN`) to predict flow rate from each stack
- Outputs predictions to CSV and visualizes results
- Supports model evaluation via checkpointed weights

---

## Folder Structure
project-root/
│
├── data/ # Folder for input videos (*.avi)
├── models/
│ └── bloodflow_cnn.py # Core 3D CNN model
├── src/
│ ├── dataset.py # Custom PyTorch Dataset
│ ├── dataloader.py # Preprocessing and loader functions
│ └── train.py # Training pipeline
│
├── outputs/
│ ├── predictions.csv # Final output with 120 predictions
│ ├── flowrate_plot.png # True vs Predicted plot
│ ├── train_loss.png # Training loss curve
│ ├── scaler.pkl # Saved target normalizer
│ └── checkpoints/ # Saved model weights
│
├── main.py # CLI entry point for training/evaluation
├── requirements.txt # Python dependencies
└── README.md # Project overview (this file)

---

## How to Use

### 1. Install dependencies

```bash
pip install -r requirements.txt
##Prepare your Data 
Place all input .avi files into the data/ folder. The expected filenames are:

0ulpermin.avi

5ulpermin.avi

50ulpermin.avi

100ulpermin.avi

150ulpermin.avi

400ulpermin.avi

Each video should be at least 5 frames long (the default sequence length).

##3. Train the model
bash
Copy
Edit
python main.py --mode train
This will:

Train BloodFlowCNN for 5 epochs

Save the best model to outputs/checkpoints/

Save a CSV of predictions to outputs/predictions.csv

Plot predicted vs. true flow rates to outputs/flowrate_plot.png

##4. Evaluate a trained model
bash
Copy
Edit
python main.py --mode evaluate --checkpoint outputs/checkpoints/best_model.pt
This will reload the model and report:

MSE (Mean Squared Error)

MAE (Mean Absolute Error)

R² Score (Explained Variance)

##Output Example (predictions.csv)
Index	TrueFlowRate	PredictedFlowRate	AbsoluteError	SquaredError	RelativeError (%)	FlowRateClass
0	400.0	380.42	19.58	383.29	4.9	High

##Requirements
nginx
Copy
Edit
torch
torchvision
opencv-python
scikit-learn
matplotlib
pandas
notebook
##Acknowledgement
Developed under the guidance of Dr. Christopher Raub and assisted by Thuc Pham at The Catholic Unniversity of America as part of a project investigating speckle-based imaging for blood flow estimation in microfluidic systems.