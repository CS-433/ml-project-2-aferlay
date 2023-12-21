

ReadMe for Road Segmentation Project

----------------------------------------------------------------------------------------------
Introduction

This repository contains the Python code for a road segmentation project using a U-Net model with a ResNet-50 encoder. The project aims to segment roads from satellite images effectively.


----------------------------------------------------------------------------------------------
Setup Instructions

Clone the Repository: Clone this repository to your local machine to get started.

Install Required Libraries: Ensure you have Python 3.x installed along with the following libraries:

torch
numpy
opencv-python (cv2)
Pillow (PIL)
segmentation_models_pytorch (smp)
pytorch_lightning
torchvision
matplotlib

You can install them using pip:
pip install torch numpy opencv-python Pillow segmentation-models-pytorch pytorch_lightning torchvision matplotlib

----------------------------------------------------------------------------------------------
Prepare the Dataset: Place your satellite images and corresponding ground truth masks in designated directories. The expected structure is:

/path_to_dataset/
├── images/          # Directory containing satellite images
└── groundtruth/     # Directory containing ground truth masks

----------------------------------------------------------------------------------------------
Running the Code

Training the Model
To train the model, use the train_and_save_model function. Customize the parameters like crop_size, batch_size, max_epochs, etc., as needed.

Example:

python
Copy code
train_and_save_model(
    images_path='path_to_images',
    groundtruth_path='path_to_groundtruth',
    crop_size=(256, 256),
    batch_size=32,
    isCropped=True,
    isResized=False,
    max_epochs=30,
    patience=5,
    step_size=68
)

----------------------------------------------------------------------------------------------
Running Model Predictions

To run predictions on the test set and save the results, use the run_model_test function. Ensure you specify the correct paths and settings.

Example:

python
Copy code
run_model_test(
    model_path="models_pytorch/model.pth",
    images_folder="path_to_test_images",
    output_folder="path_to_output",
    crop_size=(256, 256),
    step_size=88,
    title='test',
    isCropped=True,
    save_masks=True,
    plots_dir='256x256_masks_predictions/',
    binarization_threshold=0.3
)

----------------------------------------------------------------------------------------------
Reproducing Results

To replicate the results as submitted on AIcrowd, follow these steps:

Configure Paths: Update the paths in the script to match your directory structure. This includes paths for training images, ground truth masks, test images, model file, and output directories. Look for the # <===== configure comments in the script and set the paths accordingly.

Run Training (Optional): If you wish to retrain the model, run the train_and_save_model function with the appropriate parameters. This step can be skipped if you are using an already trained model.

Run Model Test: Execute the run_model_test function with the correct model path and other parameters. This will process the test images, generate predictions, and save them.

Generate Submission File: Use the provided script to convert the predicted masks into a CSV format suitable for AIcrowd submission. Make sure the mask_folder path points to where your prediction masks are stored.
Documentation

Each function in the code is well-documented, providing clear explanations of its purpose, parameters, and return values. For specific details about a function's operation and usage, refer to the inline comments in the code.

Note on Reproducibility

Environment Consistency: Ensure that the same Python environment and library versions are used as in the original setup. Differences in the environment may lead to variations in results.

Dataset Consistency: Use the same dataset as used for the AIcrowd submission. Any changes or modifications in the dataset can affect the reproducibility of the results.

Model Consistency: If using a pre-trained model, ensure it is the same model (and model state) that was used for the best AIcrowd submission.

By carefully following these instructions and ensuring consistency in the environment, dataset, and model, you should be able to reproduce the training, testing, and evaluation steps as performed in this project.
