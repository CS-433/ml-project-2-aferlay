"""
Details of the file:

This script implements a full pipeline for binary road segmentation using a U-Net architecture. 
The key components include:

1. Data Loading:
   - Images and corresponding ground truth masks are loaded from specified directories.
   - Masks are binarized to ensure consistency for binary segmentation.

2. Data Preprocessing:
   - Splits the dataset into training, validation, and test sets using an 80-10-10 ratio.

3. Model Configuration:
   - Builds a U-Net model with customizable hyperparameters such as number of layers, filters, dropout rate, and L2 regularization.
   - Uses the Adam optimizer with a learning rate of \( 1 \times 10^{-4} \).

4. Model Training:
   - Trains the U-Net model with early stopping and model checkpointing to save the best model based on validation loss.

5. Prediction and Post-processing:
   - Processes test images of size \( 608 \times 608 \) by extracting patches.
   - Applies the trained model to generate predicted binary masks using an optimized threshold.
   - Reconstructs masks from patches and saves them to the specified output folder.

6. Submission Generation:
   - Converts predicted masks into a submission-ready format (CSV) for evaluation.
"""
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers.legacy import Adam
from utils import extract_patches_from_large_image, reconstruct_image_from_patches
from helpings import*
import glob
from sklearn.model_selection import train_test_split

#########################################
# Data Loading
#########################################

print("Data Loading ...")
image_dir = "ML_course-main/projects/project2/data/training/patches/images" # Ajust
mask_dir = "ML_course-main/projects/project2/data/training/patches/groundtruth" # Ajust

image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

# Load all images into memory (if dataset fits into memory)
# If not, consider using a tf.data pipeline with lazy loading.
images = [plt.imread(f) for f in image_files]  # shape ~ (256,256,3)
masks = [plt.imread(fm) for fm in mask_files]  # shape ~ (256,256), binary mask
images = np.stack(images, axis=0)
masks = np.stack(masks, axis=0)

# Ensure masks are binary 0/1
masks = (masks > 0.5).astype(np.float32)

print("Data Load")

#########################################
# Train/Val/Test Split
#########################################
trainval_images, test_images, trainval_masks, test_masks = train_test_split(
    images, masks, test_size=0.1, random_state=42
)
train_images, val_images, train_masks, val_masks = train_test_split(
    trainval_images, trainval_masks, test_size=0.2, random_state=42
)

#########################################
# Hyperparamètres
#########################################
input_shape = train_images.shape[1:]  # Dynamically get input shape
lr = 1e-4
patience = 5
batch = 64
n_layers = 2
epochs = 2
init_filters = 8
reg = 1e-6
d_out=0.1

#########################################
# Model architecture
#########################################
model = build_unet(input_shape=input_shape,num_layers=n_layers,initial_filters=init_filters,dropout_rate=d_out,l2_reg=reg)

# Compile the model
model.compile(optimizer=Adam(learning_rate=lr), 
              loss=combined_loss,
              metrics=["accuracy"])

# Summary
model.summary()

# Callbacks
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "model.h5", save_best_only=True, monitor="val_loss", mode="min"
)
earlystop_cb = tf.keras.callbacks.EarlyStopping(
    patience=patience, restore_best_weights=True, monitor="val_loss", mode="min"
)

#########################################
# Model training and saving
#########################################

history = model.fit(
    train_images, train_masks,
    validation_data=(val_images, val_masks),
    epochs=epochs,
    batch_size=batch,
    callbacks=[checkpoint_cb, earlystop_cb]
)

# Sauvegarder le modèle final
model.save("model.h5")


###################################################################################################
#                               Test the model on a new test set 608x608                          #
###################################################################################################


#########################################
# Paths to your folders
#########################################

root_dir = "ML_course-main/projects/project2/data/test_set_images" # path to your test set images
checkpoint_path = "/Users/aurelien/ML/model.h5"  # Pre-trained model checkpoint
output_folder = "ML_course-main/projects/project2/predicted_masks"  # Output folder for masks
output_filename = "submission.csv" # Submission file name

# Load test images
print("Loading test images...")
test_imgs = get_images_test(root_dir, num_images=50)  # Adjust the number of test images if needed
img_size = 608  # Original test image size <============================== ADJUST
patch_size = 256  # Patch size 256 or 400  <============================== ADJUST
foreground_threshold = 0.42 #Ajust the threshold

# Load pre-trained model
print("Loading pre-trained model...")
model = load_model(checkpoint_path, compile=False)

# Process each test image
masks = []
print("Processing test images...")
for img in test_imgs:
    binarized_mask = predict_on_large_image(model, img, foreground_threshold, patch_size=patch_size)
    masks.append(binarized_mask)

# Save all masks to the output folder
print("Saving predicted masks to folder...")
save_predicted_masks(output_folder, masks)

# Save predictions into submission file
print("Saving predictions to submission file...")
masks_to_submission(output_filename, masks,threshold=foreground_threshold)

print(f"Predicted masks saved to folder: {output_folder}")
print(f"Submission file saved: {output_filename}")