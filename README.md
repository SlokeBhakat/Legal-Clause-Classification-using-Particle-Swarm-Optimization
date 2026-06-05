# Legal Clause Classification
## Overview
a machine learning pipeline designed to automatically classify legal clauses from contract text into 100 distinct categories (such as "Confidentiality", "Governing Laws", "Taxes", etc.); aims to streamline the analysis of large legal documents by automating clause categorization.
## Methodology
The current implementation serves as a highly efficient baseline prior to the deployment of complex deep learning architectures. It leverages:
- **TF-IDF Vectorization:** For high-dimensional, sparse text feature extraction.
- **Logistic Regression:** A robust linear classifier optimized for high-dimensional text data.
- **Particle Swarm Optimization (PSO):** A metaheuristic algorithm used for hyperparameter tuning. The PSO algorithm efficiently searches for the optimal TF-IDF vocabulary size (`max_features`) and Logistic Regression regularization strength (`C`).
- **Macro F1 Evaluation:** Given the heavily imbalanced nature of the dataset, the model is evaluated using the Macro F1 score to ensure equal performance representation across all classes, regardless of frequency.
## Dataset
The model is trained on the **LEDGAR** dataset (`coastalchp/ledgar`) hosted on Hugging Face. The dataset comprises 60,000 training samples across 100 distinct legal provisions. Due to extreme class imbalance, the pipeline computes and enforces balanced class weights during the training process.
## Repository Structure
- `data/`: Contains static configurations, including generated `class_weights.json` to handle dataset imbalance, and dataset metadata.
- `data_processing/`: Contains exploratory data analysis (EDA), data visualization, and class weight computation (`data_explore.ipynb`).
- `model_training/`: The core pipeline notebooks:
  - `logistic_regression.ipynb`: The baseline training and evaluation script.
  - `final_model_with_pso.ipynb`: The end-to-end production pipeline incorporating PSO hyperparameter tuning and final evaluation.
- `model/`: The deployment directory where finalized artifacts (the `.pkl` vectorizer, the `.pkl` model, and best parameters) are saved.
## Installation & Usage
1. Python 3.8+ installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. To run the full training pipeline and reproduce the optimized results, execute the Jupyter Notebook:
   ```bash
   jupyter notebook model_training/final_model_with_pso.ipynb
   ```
