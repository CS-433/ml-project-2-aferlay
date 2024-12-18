# Project Road Segmentation

For this task, we use a set of satellite images acquired from Google Maps, along with corresponding ground-truth images where each pixel is labeled as road (1) or background (0). The goal is to train a model (based on a U-Net architecture) to perform binary road segmentation on these images.

## Contents

- [Dataset Setup](#dataset-setup)
- [Environment & Dependencies](#environment--dependencies)
- [Usage Instructions](#usage-instructions)
- [Model Architecture & Training Details](#model-architecture--training-details)
- [Evaluation Metric](#evaluation-metric)
- [Submission Format](#submission-format)
- [Troubleshooting & Tips](#troubleshooting--tips)

---

## Dataset Setup

1. **Data Source**: The dataset is available from the [AICrowd page](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation).  
   Download the training and test datasets from there.

2. **Organizing the Data**:
   - Extract the training images and their corresponding ground-truth masks into the following folder structure:
     ```
     ML_course-main/projects/project2/data/
     ├── training/
     │   ├── images/            # Original training images
     │   ├── groundtruth/       # Original ground-truth masks
     │   └── patches/
     │       ├── images/        # Pre-processed image patches
     │       └── groundtruth/   # Pre-processed mask patches
     └── test_set_images/       # Test images (608x608), as provided
     ```

   **Note**:  
   - The code expects pre-extracted patches (`256x256` or `400x400`) in `training/patches/images` and `training/patches/groundtruth`.  
   - Test images should remain in their original `608x608` size.

---

## Environment & Dependencies

- **Python Version**: 3.7+
- **Key Libraries**:
  - `tensorflow` (with Keras)
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `Pillow`
  - `glob`, `os`

## Usage Instructions

1. **Run the Training Pipeline**:  
   - Ensure the dataset is organized correctly (as described earlier).  
   - Execute the `run.py` script to train the U-Net model:  
     ```bash
     python run.py
     ```
   - During training:
     - The model checkpoints will be saved in the current working directory.
     - Early stopping ensures the best model is selected based on validation loss.

2. **Generate Predictions on Test Images**:  
   - Modify the paths for `test_set_images` and the pre-trained model in `run.py` if necessary.
   - Predictions will be saved as masks in the `predicted_masks/` folder.
   - A CSV submission file will be generated in the root directory.

3. **Output Directory**:  
   - Predicted masks: `ML_course-main/projects/project2/predicted_masks/`  
   - Submission file: `submission_400B16F16RD.csv`  

---

## Model Architecture & Training Details

The pipeline follows a **U-Net architecture** designed for binary road segmentation.

1. **Model Hyperparameters**:  
   - Number of layers: 4  
   - Initial filters: 16  
   - Dropout rate: 0.1  
   - L2 regularization: \( 1 \times 10^{-6} \)  

2. **Optimizer**: Adam optimizer with a learning rate of \( 1 \times 10^{-4} \).  
3. **Loss Function**: Custom combined loss.  
4. **Training Setup**:  
   - Early stopping to prevent overfitting.  
   - Checkpointing to save the best-performing model.  
5. **Input/Output**:  
   - Input size: Patches of \( 256 \times 256 \) or \( 400 \times 400 \).  
   - Output: Binary mask predictions.  

---

## Evaluation Metric

The performance of the model is evaluated using the **F1 score**:  

## Submission Format

- The final predictions are saved as a **CSV file** in the following format:

- Each row corresponds to a pixel, where:
- `id` is the pixel index.  
- `prediction` is the predicted label (`0` for background, `1` for road).

- The submission file is saved as `submission_400B16F16RD.csv` in the root directory.

---

## Troubleshooting & Tips

- **Missing Data**:  
Ensure that all images and masks are present in the specified directories before running the pipeline.  

- **Memory Issues**:  
If the dataset is too large to load into memory, consider using a **data generator** or lazy loading techniques.

- **Patch Extraction**:  
If patches of size \( 256 \times 256 \) or \( 400 \times 400 \) are not available, you need to preprocess the images using a separate script.

- **Threshold Tuning**:  
The `foreground_threshold` parameter in `run.py` (default set to 0.42) can be adjusted to improve mask binarization based on the model's predictions.

- **Model Checkpoints**:  
Ensure the correct checkpoint path is provided when generating predictions on the test set.

---

## Contact

For questions, issues, or further clarifications, please refer to the [AICrowd challenge page](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation).