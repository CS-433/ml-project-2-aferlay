import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import cv2
import os
import matplotlib.image as mpimg

def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Dice Loss for binary segmentation.
    Args:
        y_true: Ground truth mask.
        y_pred: Predicted mask.
        smooth: Smoothing factor to prevent division by zero.
    Returns:
        Dice loss value.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for imbalanced data.
    Args:
        gamma: Focusing parameter to reduce the relative loss for well-classified examples.
        alpha: Balancing parameter for positive class.
    Returns:
        Focal loss function.
    """
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        weight = alpha * y_true * K.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * K.pow(y_pred, gamma)
        return K.mean(weight * cross_entropy)
    return focal_loss_fixed

def combined_loss(y_true, y_pred):
    """
    Combination of Dice Loss and Focal Loss.
    Args:
        y_true: Ground truth mask.
        y_pred: Predicted mask.
    Returns:
        Combined loss value.
    """
    dice = dice_loss(y_true, y_pred)
    focal = focal_loss(gamma=2.0, alpha=0.25)(y_true, y_pred)
    return dice + focal

def encoder_block(inputs, filters, dropout_rate=0.0, l2_reg=0.01):
    """
    Constructs an encoder block for the U-Net architecture.
    
    Args:
        inputs (tf.Tensor): 
            Input tensor with shape (batch_size, height, width, channels).
        filters (int): 
            Number of filters (feature maps) for the Conv2D layers.
        dropout_rate (float, optional): 
            Dropout rate applied after the convolutional layers. Default is 0.0.
        l2_reg (float, optional): 
            L2 regularization factor for the Conv2D kernel weights. Default is 0.01.

    Returns:
        tuple:
            - x (tf.Tensor): Output tensor after the convolution and dropout layers.
            - pool (tf.Tensor): Downsampled tensor after the MaxPooling layer.
    
    Example:
        inputs = tf.keras.layers.Input(shape=(256, 256, 3))
        x, pool = encoder_block(inputs, filters=32, dropout_rate=0.1, l2_reg=0.01)
    """
    x = tf.keras.layers.Conv2D(
        filters, (3, 3), activation="relu", kernel_initializer="he_normal",
        padding="same", kernel_regularizer=regularizers.l2(l2_reg)
    )(inputs)
    x = tf.keras.layers.Conv2D(
        filters, (3, 3), activation="relu", kernel_initializer="he_normal",
        padding="same", kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    pool = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
    return x, pool

def decoder_block(inputs, skip_connection, filters, dropout_rate=0.0, l2_reg=0.01):
    """
    Constructs a decoder block for the U-Net architecture.

    The block upsamples the input using Conv2DTranspose, concatenates it with the 
    corresponding skip connection, applies two Conv2D layers with ReLU activation, 
    and a Dropout layer for regularization.

    Args:
        inputs (tf.Tensor): 
            Input tensor from the previous layer.
        skip_connection (tf.Tensor): 
            Tensor from the corresponding encoder block for concatenation.
        filters (int): 
            Number of filters for the Conv2D layers.
        dropout_rate (float, optional): 
            Dropout rate for regularization. Default is 0.0.
        l2_reg (float, optional): 
            L2 regularization factor. Default is 0.01.

    Returns:
        tf.Tensor: Output tensor after upsampling, convolution, and dropout.
    """
    x = tf.keras.layers.Conv2DTranspose(
        filters, (2, 2), strides=(2, 2), padding="same"
    )(inputs)
    x = tf.keras.layers.concatenate([x, skip_connection])
    x = tf.keras.layers.Conv2D(
        filters, (3, 3), activation="relu", kernel_initializer="he_normal",
        padding="same", kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = tf.keras.layers.Conv2D(
        filters, (3, 3), activation="relu", kernel_initializer="he_normal",
        padding="same", kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x

def build_unet(input_shape, num_layers=None, initial_filters=16, dropout_rate=0.1, l2_reg=0.01):
    """
    Build U-Net for both multiples of 2 (e.g., 256x256) and 400x400 inputs.

    Args:
        input_shape: Shape of the input image (e.g., (256, 256, 3) or (400, 400, 3)).
        num_layers: Number of encoder/decoder layers (default: computed based on input size).
        initial_filters: Number of filters in the first encoder block.
        dropout_rate: Dropout rate for all blocks.
        l2_reg: L2 regularization factor for all convolutional layers.

    Returns:
        model: A compiled U-Net model.
    """
    height, width, _ = input_shape

    if height == width and height in [128, 256, 512]:
        # General U-Net for multiples of 2
        if num_layers is None:
            num_layers = int(tf.math.log(float(height)) / tf.math.log(2.0)) - 4  # Default layers based on size

        inputs = tf.keras.layers.Input(input_shape)
        encoder_outputs = []
        x = inputs

        # Encoding Path
        for i in range(num_layers):
            filters = initial_filters * (2 ** i)
            enc, x = encoder_block(x, filters, dropout_rate=dropout_rate + (0.1 * i), l2_reg=l2_reg)
            encoder_outputs.append(enc)

        # Bottleneck
        bottleneck, _ = encoder_block(x, initial_filters * (2 ** num_layers), dropout_rate=dropout_rate + 0.2, l2_reg=l2_reg)

        # Decoding Path
        for i in range(num_layers - 1, -1, -1):
            filters = initial_filters * (2 ** i)
            bottleneck = decoder_block(bottleneck, encoder_outputs[i], filters, dropout_rate=dropout_rate + (0.1 * i), l2_reg=l2_reg)

        # Output Layer
        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(bottleneck)

    elif height == 400 and width == 400:
        # Specialized U-Net for 400x400
        if num_layers is None:
            num_layers = 3  # Default to 3 layers for 400x400

        inputs = tf.keras.layers.Input(input_shape)
        encoder_outputs = []
        x = inputs

        # Encoding Path
        for i in range(num_layers):
            filters = initial_filters * (2 ** i)
            enc, x = encoder_block(x, filters, dropout_rate=dropout_rate + (0.1 * i), l2_reg=l2_reg)
            encoder_outputs.append(enc)

        # Bottleneck
        bottleneck, _ = encoder_block(x, initial_filters * (2 ** num_layers), dropout_rate=dropout_rate + 0.2, l2_reg=l2_reg)

        # Decoding Path
        for i in range(num_layers - 1, -1, -1):
            filters = initial_filters * (2 ** i)
            bottleneck = decoder_block(bottleneck, encoder_outputs[i], filters, dropout_rate=dropout_rate + (0.1 * i), l2_reg=l2_reg)

        # Output Layer
        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(bottleneck)

    else:
        raise ValueError("Unsupported input shape. Only multiples of 2 (e.g., 256x256) or 400x400 are supported.")

    model = tf.keras.Model(inputs, outputs, name=f"U-Net_{height}x{width}")
    return model

def plot_training_history(history):
    """
    Plots the training and validation loss and accuracy curves over epochs.

    Args:
        history (tf.keras.callbacks.History): 
            Training history object containing loss and metric values for each epoch.
    
    Displays:
        Two subplots:
        - Training and validation loss.
        - Training and validation accuracy.
    """
    plt.figure(figsize=(6, 3))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Validation Loss")
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label="Train Accuracy")
    plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
    plt.title("Accuracy over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
    
def find_optimal_threshold(model, val_images, val_masks, thresholds=np.arange(0.1, 0.91, 0.02)):
    """
    Finds the threshold that maximizes the F1-score on the validation data.

    Args:
        model: Trained model used for predictions.
        val_images: Validation images.
        val_masks: Ground truth masks for the validation images.
        thresholds: Range of thresholds to evaluate (default: 0.1 to 0.9 with step 0.02).

    Returns:
        best_threshold (float): The threshold that gives the highest F1-score.
        f1_scores (list): List of F1-scores for each threshold tested.
    """
    preds = model.predict(val_images)
    best_threshold = 0
    best_f1 = 0
    f1_scores = []

    for threshold in thresholds:
        binarized_preds = (preds > threshold).astype(np.float32)
        flat_preds = binarized_preds.flatten()
        flat_masks = val_masks.flatten()
        f1 = f1_score(flat_masks, flat_preds)
        f1_scores.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, f1_scores

def predict_on_image(model, image, threshold):
    """
    Makes a prediction on a single image using a given threshold.

    Args:
        model: Trained model.
        image: Input image (shape: (H, W, C)).
        threshold: Threshold to binarize the prediction.

    Returns:
        Binary prediction of the image.
    """ 
    pred = model.predict(image[np.newaxis, ...])
    pred_binarized = (pred > threshold).astype(np.float32)
    return pred_binarized[0]

def calculate_f1_on_image(pred_binarized, ground_truth):
    """
    Calculates the F1-score for a single image.

    Args:
        pred_binarized: Binary prediction of the image.
        ground_truth: Ground truth mask.

    Returns:
        F1-score.
    """
    flat_pred = pred_binarized.flatten()
    flat_truth = ground_truth.flatten()
    return f1_score(flat_truth, flat_pred)

def visualize_prediction(image, ground_truth, prediction):
    """
    Visualizes the input image, ground truth mask, and prediction.

    Args:
        image: Input image (H, W, C).
        ground_truth: Ground truth mask (H, W).
        prediction: Binary prediction (H, W).
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Image")

    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth, cmap="gray")
    plt.title("Ground Truth")

    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap="gray")
    plt.title("Prediction")

    plt.show()
    
    
def extract_patches_from_large_image(image, patch_size=256):
    """
    Extracts overlapping patches from a large image.

    Args:
        image: Input image (H, W, C) from which patches are extracted.
        patch_size: Size of each patch (default: 256x256).

    Returns:
        patches: List of extracted patches of size (patch_size, patch_size, C).
        coords: List of (y, x) coordinates for each patch's top-left corner.
    """
    y_indices = [0, 608 - patch_size, (608 - patch_size) // 2]  # [0, 176, 352]
    x_indices = [0, 608 - patch_size, (608 - patch_size) // 2]  # [0, 176, 352]

    patches = []
    coords = []

    for y in sorted(y_indices):
        for x in sorted(x_indices):
            patch = image[y:y + patch_size, x:x + patch_size, :]
            patches.append(patch)
            coords.append((y, x))

    return patches, coords

def reconstruct_image_from_patches(patches, coords, original_shape, patch_size=256):
    """
    Reconstructs an image from overlapping patches.

    Args:
        patches: List of extracted patches (H, W, C).
        coords: List of (y, x) coordinates corresponding to each patch.
        original_shape: Shape of the original image (H, W, C).
        patch_size: Size of each patch (default: 256x256).

    Returns:
        Reconstructed image with overlapping regions averaged.
    """
    reconstructed = np.zeros(original_shape, dtype=np.float32)
    count = np.zeros(original_shape[:2], dtype=np.float32)

    for patch, (y, x) in zip(patches, coords):
        reconstructed[y:y + patch_size, x:x + patch_size, :] += patch
        count[y:y + patch_size, x:x + patch_size] += 1

    # Normalize to handle overlapping areas
    reconstructed /= np.maximum(count[:, :, None], 1)
    return reconstructed

def predict_on_large_image(model, image, threshold, patch_size=256):
    """
    Predict a mask for a large image using the model and apply thresholding.
    
    Args:
        model: Trained U-Net model.
        image: Input image (e.g., 608x608x3).
        threshold: Optimal threshold for binarization.
        patch_size: Size of each patch for prediction.
    
    Returns:
        binarized_mask: Binary mask of the same size as the input image.
    """
    # Step 1: Extract patches from the image
    patches, coords = extract_patches_from_large_image(image, patch_size)

    # Step 2: Predict on each patch
    predicted_patches = []
    for patch in patches:
        patch = np.expand_dims(patch, axis=0)  # Add batch dimension
        prediction = model.predict(patch, verbose=0)  # Predict
        predicted_patches.append(prediction[0])  # Output shape (256, 256, 1)

    # Step 3: Reconstruct the full mask from patches
    predicted_mask = reconstruct_image_from_patches(predicted_patches, coords, image.shape[:2] + (1,), patch_size)

    # Step 4: Apply thresholding
    binarized_mask = (predicted_mask >= threshold).astype(np.uint8)

    return binarized_mask

def add_manual_grid_to_image(image, patch_size=256):
    """
    Add a grid to the image to visualize patch boundaries.
    
    Args:
        image: Original image (HxWxC).
        patch_size: Size of each patch (default: 256).
    """
    # Define indices for grid lines
    y_indices = [0, (608 - patch_size) // 2, 608 - patch_size]  # [0, 176, 352]
    print(y_indices)
    x_indices = [0, (608 - patch_size) // 2, 608 - patch_size]  # [0, 176, 352]

    plt.imshow(image.astype(np.uint8))

    # Add horizontal lines
    for y in y_indices:
        plt.axhline(y, color="red", linestyle="--", linewidth=1.5)
        plt.axhline(y + patch_size, color="blue", linestyle="--", linewidth=1.5)

    # Add vertical lines
    for x in x_indices:
        plt.axvline(x, color="red", linestyle="--", linewidth=1.5)
        plt.axvline(x + patch_size, color="blue", linestyle="--", linewidth=1.5)

    plt.title("Image with Manual Grid Overlay")
    plt.axis("off")
    plt.show()

def visualize_prediction_with_overlay(image, predicted_mask, patch_coords, patch_size=256):
    """
    Overlay the predicted mask on the original image with a transparent red overlay,
    and add a grid to visualize patch boundaries based on the patch coordinates.

    Args:
        image: Original image (shape: HxWx3, values between 0-255).
        predicted_mask: Binary mask (shape: HxW, values 0 or 1).
        patch_coords: Coordinates of patches used during prediction.
        patch_size: Size of the patches used for prediction.
    """
    # Convert the image to float for manipulation
    image_float = image

    # Create a red overlay
    overlay = np.zeros_like(image_float)
    overlay[:, :, 0] = predicted_mask.squeeze()  # Red channel only

    # Blend the two images (add transparency)
    alpha = 0.3  # Control transparency
    blended = (1 - alpha) * image_float + alpha * overlay

    # Display the result with grid
    plt.figure(figsize=(12, 6))
    plt.title("Predicted Mask Overlay with Grid")
    plt.imshow(blended)


    # Define indices for grid lines
    y_indices = [0, (608 - patch_size) // 2, 608 - patch_size]  # [0, 176, 352]
    print(y_indices)
    x_indices = [0, (608 - patch_size) // 2, 608 - patch_size]  # [0, 176, 352]
    # Add horizontal lines
    for y in y_indices:
        plt.axhline(y, color="red", linestyle="--", linewidth=1.5)
        plt.axhline(y + patch_size, color="blue", linestyle="--", linewidth=1.5)

    # Add vertical lines
    for x in x_indices:
        plt.axvline(x, color="red", linestyle="--", linewidth=1.5)
        plt.axvline(x + patch_size, color="blue", linestyle="--", linewidth=1.5)

    plt.axis("off")
    plt.show()
    
# assign a label to a patch
def patch_to_label(patch,threshold):
    """
    Converts a patch to a label based on the mean value of the patch.

    Parameters:
    patch (numpy.ndarray): The patch to be converted to a label.

    Returns:
    int: The label of the patch. Returns 1 if the mean value of the patch is greater than foreground_threshold, otherwise returns 0.
    """
    df = np.mean(patch)
    if df > threshold:
        return 1
    else:
        return 0

def get_images_test(root_dir, num_images=50):
    """
    Extract test images from directories where each image is stored in a separate folder.

    Args:
        root_dir (str): Path to the root directory containing subfolders for test images.
        num_images (int): Number of test images to load (default is 50).

    Returns:
        tf.Tensor: A tensor containing all the loaded test images.
    """
    imgs = []

    # Iterate over the expected test image folders
    for i in range(1, num_images + 1):
        image_folder = os.path.join(root_dir, f"test_{i}")  # e.g., test_1, test_2
        image_filename = os.path.join(image_folder, f"test_{i}.png")  # e.g., test_1/test_1.png

        if os.path.isfile(image_filename):
            print(f"Loading {image_filename}")
            img = mpimg.imread(image_filename)  # Load the image
            imgs.append(img)
        else:
            print(f"File {image_filename} does not exist")

    if not imgs:
        raise FileNotFoundError("No images were loaded. Check the directory structure or filenames.")

    imgs = tf.stack(imgs)  # Stack images into a single tensor
    print(f"Loaded {len(imgs)} images.")
    return imgs
def mask_to_submission_strings(im,im_number,threshold):
    """
    Reads a single image and outputs the strings that should go into the submission file
    Used in the masks_to_submission function
    """
    patch_size = 16
    for j in range(0, im.shape
                   [1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch,threshold)
            yield("{:03d}_{}_{},{}".format(im_number, j, i, label))


def masks_to_submission(submission_filename, imgs,threshold):
    """
    Converts predicted masks into a submission file
    Submission filename: .csv type
    imgs: list of all predicted masks
    Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for img_num in range(len(imgs)):
          f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(imgs[img_num],img_num+1,threshold))

def predict_on_large_image(model, image, threshold, patch_size=256):
    """
    Predict a mask for a large image using the model and apply thresholding.
    
    Args:
        model: Trained U-Net model.
        image: Input image (e.g., 608x608x3).
        threshold: Optimal threshold for binarization.
        patch_size: Size of each patch for prediction.
    
    Returns:
        binarized_mask: Binary mask of the same size as the input image.
    """
    # Step 1: Extract patches from the image
    patches, coords = extract_patches_from_large_image(image, patch_size)

    # Step 2: Predict on each patch
    predicted_patches = []
    for patch in patches:
        patch = np.expand_dims(patch, axis=0)  # Add batch dimension
        prediction = model.predict(patch, verbose=0)  # Predict
        predicted_patches.append(prediction[0])  # Output shape (256, 256, 1)

    # Step 3: Reconstruct the full mask from patches
    predicted_mask = reconstruct_image_from_patches(predicted_patches, coords, image.shape[:2] + (1,), patch_size)

    # Step 4: Apply thresholding
    binarized_mask = (predicted_mask >= threshold).astype(np.uint8)

    return binarized_mask
def save_predicted_masks(output_folder, reconstructed_masks):
    """
    Saves the predicted masks as image files in the specified output folder.

    Args:
        output_folder (str): Path to the folder where masks will be saved.
        reconstructed_masks (list): List of predicted masks as numpy arrays.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    for i, mask in enumerate(reconstructed_masks):
        # Convert mask to uint8 format for saving
        mask_uint8 = (mask.squeeze() * 255).astype(np.uint8)
        
        # Define filename and save using OpenCV
        filename = os.path.join(output_folder, f"mask_{i+1}.png")
        cv2.imwrite(filename, mask_uint8)
        print(f"Saved: {filename}")
        
def calculate_f1_on_set(model, test_images, test_masks, threshold=0.5):
    """
    Calculate the average F1-score on the test set, including empty masks.

    Args:
        model: Trained model.
        test_images: Numpy array of test images (shape: N, H, W, C).
        test_masks: Numpy array of ground truth masks (shape: N, H, W).
        threshold: Threshold to binarize predictions.

    Returns:
        average_f1: Average F1-score over the test set.
    """
    all_f1_scores = []

    for img, true_mask in zip(test_images, test_masks):
        # Expand dimensions for prediction
        img_input = np.expand_dims(img, axis=0)
        
        # Model prediction
        pred_mask = model.predict(img_input, verbose=0)[0, :, :, 0]
        
        # Binarize prediction
        pred_mask_bin = (pred_mask >= threshold).astype(np.uint8)
        
        # Flatten masks
        true_mask_flat = true_mask.flatten()
        pred_mask_bin_flat = pred_mask_bin.flatten()
        
        # Calculate F1-score, including zero_division for empty masks
        f1 = f1_score(true_mask_flat, pred_mask_bin_flat, zero_division=1)
        all_f1_scores.append(f1)

    # Compute average F1-score
    average_f1 = np.mean(all_f1_scores) if all_f1_scores else 0.0
    print(f"Average F1-score on set: {average_f1:.4f}")
    return average_f1