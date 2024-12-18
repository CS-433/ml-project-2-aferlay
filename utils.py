import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from scipy.ndimage import rotate, shift, zoom
import random


def load_image(filepath):
    """
    Load an image from the specified file path.
    
    Args:
        filepath (str): Path to the image file.
    
    Returns:
        numpy.ndarray: Loaded image as a NumPy array.
    """
    return mpimg.imread(filepath)

def display_image_pair(img, gt_img, title="Image and Ground Truth"):
    """
    Display a satellite image alongside its ground truth mask.
    
    Args:
        img (numpy.ndarray): Input satellite image.
        gt_img (numpy.ndarray): Corresponding ground truth binary mask.
        title (str, optional): Title for the figure. Default is "Image and Ground Truth".
    
    Returns:
        None: Displays the image and mask side-by-side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(img)
    axes[0].set_title("Satellite Image")
    axes[0].axis("off")
    axes[1].imshow(gt_img, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")
    fig.suptitle(title)
    plt.show()

def display_comparison(original_img, original_mask, transformed_img, transformed_mask, title="Augmentation Check"):
    """
    Display original and transformed images with their corresponding masks.
    
    Args:
        original_img (numpy.ndarray): Original input image.
        original_mask (numpy.ndarray): Original ground truth mask.
        transformed_img (numpy.ndarray): Transformed (augmented) input image.
        transformed_mask (numpy.ndarray): Transformed ground truth mask.
        title (str, optional): Title for the figure. Default is "Augmentation Check".
    
    Returns:
        None: Displays a 2x2 grid comparing original and transformed images/masks.
    """
    fig, axes = plt.subplots(2, 2, figsize=(4, 4))
    
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(original_mask, cmap="gray")
    axes[0, 1].set_title("Original Mask")
    axes[0, 1].axis("off")
    
    axes[1, 0].imshow(transformed_img)
    axes[1, 0].set_title("Transformed Image")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(transformed_mask, cmap="gray")
    axes[1, 1].set_title("Transformed Mask")
    axes[1, 1].axis("off")
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    
def save_image(arr, filepath):
    """
    Save an image array to a file.

    Args:
        arr (numpy.ndarray): Image array, either in float [0, 1] or integer format.
        filepath (str): Path to save the image file.

    Returns:
        None: Saves the image to the specified file path.
    """
    # Convert to uint8 if necessary
    if arr.max() <= 1.0:
        arr_uint8 = (arr * 255).astype(np.uint8)
    else:
        arr_uint8 = arr.astype(np.uint8)
    Image.fromarray(arr_uint8).save(filepath)

def img_float_to_uint8(img):
    """
    Convert a float image to uint8 format.

    Args:
        img (numpy.ndarray): Image with pixel values in float format.

    Returns:
        numpy.ndarray: Image converted to uint8 format with values in [0, 255].
    """
    rimg = img - np.min(img)
    return (rimg / np.max(rimg) * 255).round().astype(np.uint8)

def make_img_overlay(img, predicted_img):
    """
    Create an overlay of the input image and the predicted binary mask.

    Args:
        img (numpy.ndarray): Original input image in float format.
        predicted_img (numpy.ndarray): Predicted binary mask with values 0 or 1.

    Returns:
        PIL.Image: Blended image combining the original input and the mask overlay.
    """
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    # Assuming predicted_img is binary: 0 or 1
    color_mask[:, :, 0] = (predicted_img * 255).astype(np.uint8)  # Red for the mask

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, "RGB").convert("RGBA")
    overlay = Image.fromarray(color_mask, "RGB").convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

#################################
# Individual Transformations
#################################

def random_rotate(image, mask):
    """
    Apply a random rotation to both the image and its corresponding mask.

    Args:
        image (numpy.ndarray): Input image with shape (H, W, C).
        mask (numpy.ndarray): Binary or grayscale mask with shape (H, W).

    Returns:
        tuple: Rotated image and mask with the same dimensions as the input.
    """
    angle = np.random.uniform(-30, 30)  # Random angle between -30 and 30 degrees
    rot_img = rotate(image, angle, reshape=False, order=1)  # Bilinear interpolation
    rot_mask = rotate(mask, angle, reshape=False, order=0)  # Nearest neighbor interpolation
    return rot_img, rot_mask

def random_translation(image, mask):
    """
    Apply a random translation (shift) to both the image and its corresponding mask.

    Args:
        image (numpy.ndarray): Input image with shape (H, W, C).
        mask (numpy.ndarray): Binary or grayscale mask with shape (H, W).

    Returns:
        tuple: Translated image and mask shifted by a random number of pixels.
    """
    shift_y = np.random.randint(-10, 11)  # Random shift in y-direction (-10 to 10 pixels)
    shift_x = np.random.randint(-10, 11)  # Random shift in x-direction (-10 to 10 pixels)
    translated_img = shift(image, shift=(shift_y, shift_x, 0), order=1)  # Bilinear interpolation
    translated_mask = shift(mask, shift=(shift_y, shift_x), order=0)  # Nearest neighbor interpolation
    return translated_img, translated_mask

def random_zoom(image, mask):
    """
    Apply a random zoom to both the image and its corresponding mask, with resizing to the original shape.

    Args:
        image (numpy.ndarray): Input image with shape (H, W, C).
        mask (numpy.ndarray): Binary or grayscale mask with shape (H, W).

    Returns:
        tuple: Zoomed image and mask resized to match the original dimensions.
    """
    z_factor = np.random.uniform(0.8, 1.2)  # Zoom factor between 0.8 (zoom out) and 1.2 (zoom in)
    zoomed_img = zoom(image, (z_factor, z_factor, 1), order=1)  # Bilinear interpolation
    zoomed_mask = zoom(mask, (z_factor, z_factor), order=0)  # Nearest neighbor interpolation

    # Crop or pad the image/mask to match original size
    h, w, _ = image.shape
    zh, zw, _ = zoomed_img.shape

    if zh >= h and zw >= w:
        # Crop if zoomed in
        start_h = (zh - h) // 2
        start_w = (zw - w) // 2
        final_img = zoomed_img[start_h:start_h+h, start_w:start_w+w, :]
        final_mask = zoomed_mask[start_h:start_h+h, start_w:start_w+w]
    else:
        # Pad if zoomed out
        final_img = np.zeros((h, w, 3), dtype=zoomed_img.dtype)
        final_mask = np.zeros((h, w), dtype=zoomed_mask.dtype)
        pad_h = (h - zh) // 2
        pad_w = (w - zw) // 2
        final_img[pad_h:pad_h+zh, pad_w:pad_w+zw, :] = zoomed_img
        final_mask[pad_h:pad_h+zh, pad_w:pad_w+zw] = zoomed_mask

    return final_img, final_mask

#################################
# Combined Random Augmentation
#################################
def random_augmentation(image, mask,
                        rotate_prob=0.5,
                        translate_prob=0.5,
                        zoom_prob=0.5,
                        flip_prob=0.5):
    """
    Apply random augmentations to an image and its corresponding mask, including rotation, 
    translation, zoom, and horizontal flipping, based on specified probabilities.

    Args:
        image (numpy.ndarray): Input image with shape (H, W, C).
        mask (numpy.ndarray): Binary or grayscale mask with shape (H, W).
        rotate_prob (float, optional): Probability of applying rotation. Default is 0.5.
        translate_prob (float, optional): Probability of applying translation. Default is 0.5.
        zoom_prob (float, optional): Probability of applying zoom. Default is 0.5.
        flip_prob (float, optional): Probability of applying horizontal flip. Default is 0.5.

    Returns:
        tuple: Augmented image and mask with the same dimensions as the input.
    """
    img, msk = image, mask

    # Random rotation with probability
    if np.random.rand() < rotate_prob:
        img, msk = random_rotate(img, msk)

    # Random translation with probability
    if np.random.rand() < translate_prob:
        img, msk = random_translation(img, msk)

    # Random zoom with probability
    if np.random.rand() < zoom_prob:
        img, msk = random_zoom(img, msk)

    # Random horizontal flip with probability
    if np.random.rand() < flip_prob:
        img = img[:, ::-1, :]  # Flip horizontally
        msk = msk[:, ::-1]

    return img, msk


def random_verification(augmented_img_dir, augmented_gt_dir, n=3):
    """
    Randomly select and visualize n images from the augmented dataset with their corresponding masks overlaid.

    Args:
        augmented_img_dir (str): Directory containing the augmented images.
        augmented_gt_dir (str): Directory containing the corresponding ground truth masks.
        n (int, optional): Number of images to visualize. Default is 3.

    Returns:
        None: Displays a matplotlib figure showing the selected images with overlaid masks.
    """
    all_aug_images = os.listdir(augmented_img_dir)
    all_aug_images = [f for f in all_aug_images if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if len(all_aug_images) == 0:
        print("No images found in the augmented directory.")
        return

    # Randomly choose n images
    selected_images = random.sample(all_aug_images, min(n, len(all_aug_images)))

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]  # Ensure axes is iterable when n=1

    for i, ax in enumerate(axes):
        img_file = selected_images[i]
        img_path = os.path.join(augmented_img_dir, img_file)

        # The mask should have the same filename in the ground truth folder
        mask_path = os.path.join(augmented_gt_dir, img_file)

        if not os.path.exists(mask_path):
            ax.set_title("No matching mask found")
            ax.axis("off")
            continue

        img = load_image(img_path)
        msk = load_image(mask_path)

        # Create an overlay of the image and mask
        overlayed = make_img_overlay(img, msk)

        ax.imshow(overlayed)
        ax.set_title(f"Verification {i+1}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def img_float_to_uint8(img):
    """
    Convert a float image to uint8 format.

    Args:
        img (numpy.ndarray): Input image with pixel values in float format.

    Returns:
        numpy.ndarray: Image converted to uint8 format with values scaled to [0, 255].
    """
    rimg = img - np.min(img)
    return (rimg / np.max(rimg) * 255).round().astype(np.uint8)
def make_img_overlay(img, predicted_img):
    """
    Create an overlay of the input image and the predicted binary mask.

    Args:
        img (numpy.ndarray): Original input image in float format.
        predicted_img (numpy.ndarray): Predicted binary mask with values 0 or 1.

    Returns:
        PIL.Image: Blended image combining the input image and mask overlay.
    """
    w, h = img.shape[:2]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = (predicted_img * 255).astype(np.uint8)  # Red for the mask

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, "RGB").convert("RGBA")
    overlay = Image.fromarray(color_mask, "RGB").convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


def translate_fixed(image, mask, shift_y=10, shift_x=5):
    """
    Apply a fixed translation to both the image and mask.

    Args:
        image (numpy.ndarray): Input image with shape (H, W, C).
        mask (numpy.ndarray): Corresponding mask with shape (H, W).
        shift_y (int, optional): Vertical shift. Default is 10.
        shift_x (int, optional): Horizontal shift. Default is 5.

    Returns:
        tuple: Translated image and mask.
    """
    translated_img = shift(image, shift=(shift_y, shift_x, 0), order=1)
    translated_mask = shift(mask, shift=(shift_y, shift_x), order=0)
    return translated_img, translated_mask


def rotate_fixed(image, mask, angle=20):
    """
    Apply a fixed rotation to both the image and mask.

    Args:
        image (numpy.ndarray): Input image with shape (H, W, C).
        mask (numpy.ndarray): Corresponding mask with shape (H, W).
        angle (float, optional): Rotation angle in degrees. Default is 20.

    Returns:
        tuple: Rotated image and mask.
    """
    rot_img = rotate(image, angle, reshape=False, order=1)
    rot_mask = rotate(mask, angle, reshape=False, order=0)
    return rot_img, rot_mask


def zoom_fixed(image, mask, zoom_factor=1.2):
    """
    Apply a fixed zoom transformation to both the image and mask.

    Args:
        image (numpy.ndarray): Input image with shape (H, W, C).
        mask (numpy.ndarray): Corresponding mask with shape (H, W).
        zoom_factor (float, optional): Zoom scaling factor. Default is 1.2.

    Returns:
        tuple: Zoomed image and mask resized to original dimensions.
    """
    zoomed_img = zoom(image, (zoom_factor, zoom_factor, 1), order=1)
    zoomed_mask = zoom(mask, (zoom_factor, zoom_factor), order=0)

    # Crop or pad to original size
    h, w, _ = image.shape
    zh, zw, _ = zoomed_img.shape
    if zh >= h and zw >= w:
        # Crop if zoomed in
        start_h = (zh - h) // 2
        start_w = (zw - w) // 2
        final_img = zoomed_img[start_h:start_h+h, start_w:start_w+w, :]
        final_mask = zoomed_mask[start_h:start_h+h, start_w:start_w+w]
    else:
        # Pad if zoomed out
        final_img = np.zeros((h, w, 3), dtype=zoomed_img.dtype)
        final_mask = np.zeros((h, w), dtype=zoomed_mask.dtype)
        pad_h = (h - zh) // 2
        pad_w = (w - zw) // 2
        final_img[pad_h:pad_h+zh, pad_w:pad_w+zw, :] = zoomed_img
        final_mask[pad_h:pad_h+zh, pad_w:pad_w+zw] = zoomed_mask

    return final_img, final_mask


def flip_horizontal(image, mask):
    """
    Apply a horizontal flip to both the image and mask.

    Args:
        image (numpy.ndarray): Input image with shape (H, W, C).
        mask (numpy.ndarray): Corresponding mask with shape (H, W).

    Returns:
        tuple: Horizontally flipped image and mask.
    """
    flipped_img = image[:, ::-1, :]
    flipped_mask = mask[:, ::-1]
    return flipped_img, flipped_mask


def multiple_ops(image, mask):
    """
    Apply a sequence of fixed augmentations: rotation, translation, zoom, and flip.

    Args:
        image (numpy.ndarray): Input image with shape (H, W, C).
        mask (numpy.ndarray): Corresponding mask with shape (H, W).

    Returns:
        tuple: Augmented image and mask after applying multiple operations.
    """
    img, msk = rotate_fixed(image, mask, angle=15)
    img, msk = translate_fixed(img, msk, shift_y=-8, shift_x=8)
    img, msk = zoom_fixed(img, msk, zoom_factor=1.1)
    img, msk = flip_horizontal(img, msk)
    return img, msk

def show_transformations(original_img, original_mask):
    # Apply each transformation
    trans_img, trans_mask = translate_fixed(original_img, original_mask)
    rot_img, rot_mask = rotate_fixed(original_img, original_mask, angle=20)
    zm_img, zm_mask = zoom_fixed(original_img, original_mask, zoom_factor=1.2)
    flip_img, flip_mask = flip_horizontal(original_img, original_mask)
    multi_img, multi_mask = multiple_ops(original_img, original_mask)
    
    # Create overlays for visualization
    orig_overlay = make_img_overlay(original_img, original_mask)
    trans_overlay = make_img_overlay(trans_img, trans_mask)
    rot_overlay = make_img_overlay(rot_img, rot_mask)
    zm_overlay = make_img_overlay(zm_img, zm_mask)
    flip_overlay = make_img_overlay(flip_img, flip_mask)
    multi_overlay = make_img_overlay(multi_img, multi_mask)
    
    # Create a 3x2 grid for the subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    
    # Row 1: a, b
    axes[0, 0].imshow(orig_overlay)
    axes[0, 0].set_title("(a) Original", fontsize=16)
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(trans_overlay)
    axes[0, 1].set_title("(b) Translation", fontsize=16)
    axes[0, 1].axis("off")
    
    # Row 2: c, d
    axes[1, 0].imshow(rot_overlay)
    axes[1, 0].set_title("(c) Rotation 20°", fontsize=16)
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(zm_overlay)
    axes[1, 1].set_title("(d) Zoom 1.2x", fontsize=16)
    axes[1, 1].axis("off")
    
    # Row 3: e, f
    axes[2, 0].imshow(flip_overlay)
    axes[2, 0].set_title("(e) Horizontal Flip", fontsize=16)
    axes[2, 0].axis("off")
    
    axes[2, 1].imshow(multi_overlay)
    axes[2, 1].set_title("(f) Multiple Ops", fontsize=16)
    axes[2, 1].axis("off")
    
    # Adjust layout
    plt.tight_layout()
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


def normalize_image(img):
    """
    Normalize the image to the [0,1] range.
    If the image is already in float and close to this range, 
    this step may be optional, but it's good practice to ensure normalization.
    """
    # If the image is uint8 (0-255), convert to float and normalize:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    # If the image is already float (e.g., from a PNG), ensure it's within [0,1].
    # Here we just clip in case of floating values slightly out of range:
    return np.clip(img, 0.0, 1.0)

def extract_patches(img, gt, patch_size=256):
    """
    Extract four overlapping 256x256 patches from the 400x400 image and ground truth.
    
    Since we have 400x400 images, we create patches:
    - Patch 1: top-left corner (0,0) to (256,256)
    - Patch 2: top-left corner (0,144) to (256,400)
    - Patch 3: top-left corner (144,0) to (400,256)
    - Patch 4: top-left corner (144,144) to (400,400)
    
    This way, every pixel of the original image is included in at least one patch.
    """
    # Define the start points for patch extraction
    starts = [0, 400 - patch_size]  # [0, 144] for start indices
    patches_img = []
    patches_gt = []
    
    for row_start in starts:
        for col_start in starts:
            # Extract patch from the image
            img_patch = img[row_start:row_start+patch_size, col_start:col_start+patch_size, :]
            # Extract patch from the ground truth (assumed single channel)
            gt_patch = gt[row_start:row_start+patch_size, col_start:col_start+patch_size]
            
            patches_img.append(img_patch)
            patches_gt.append(gt_patch)
    
    return patches_img, patches_gt
def check_image_dimensions(directory):
    """
    Check if all images in a directory have the same dimensions.
    
    Args:
        directory: Path to the directory containing images.
    
    Returns:
        A dictionary summarizing the dimensions of all images.
    """
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))]
    
    if not image_files:
        print("No images found in the directory.")
        return None
    
    dimensions = {}
    
    for img_file in image_files:
        img_path = os.path.join(directory, img_file)
        with Image.open(img_path) as img:
            dim = img.size  # (width, height)
            if dim in dimensions:
                dimensions[dim].append(img_file)
            else:
                dimensions[dim] = [img_file]
    
    return dimensions

def summarize_dimensions(dimensions_dict):
    """
    Print a summary of dimensions for all images.
    
    Args:
        dimensions_dict: Dictionary with dimensions as keys and list of images as values.
    """
    if not dimensions_dict:
        print("No dimensions to summarize.")
        return
    
    print("Image Dimensions Summary:")
    for dim, files in dimensions_dict.items():
        print(f"Dimension: {dim}, Count: {len(files)}")
        if len(files) < 10:  # Show file names if fewer than 10
            print(f"Files: {files}")
        print()