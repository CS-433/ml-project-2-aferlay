import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from PIL import Image
import cv2
import segmentation_models_pytorch as smp
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re


def load_images_from_folder(folder, is_mask=False):
    """
    Load images from a specified folder and return them as a list of numpy arrays.

    This function iterates over all files in the given folder, loading each image file. 
    It supports both normal RGB images and mask images. For mask images, they are 
    converted to grayscale. All images are normalized by dividing pixel values by 255.

    Parameters:
    - folder (str): The path to the folder containing image files.
    - is_mask (bool, optional): If set to True, images are loaded as grayscale (used for mask images). 
                                Defaults to False.

    Returns:
    - list: A list of images represented as numpy arrays.
    - list: A list of filenames corresponding to each loaded image.
    """
    images = []
    filenames = []

    # Iterate through each file in the folder
    for filename in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, filename)

        # Check if the file path points to a file
        if os.path.isfile(img_path):
            # Load the image as grayscale if it's a mask, otherwise load as RGB
            img = Image.open(img_path).convert('L' if is_mask else 'RGB')
            
            # Normalize the image and add it to the list
            img = np.array(img) / 255.0
            images.append(img)
            filenames.append(filename)

    return images, filenames

def resize_image(image, target_size=(256, 256)):
    """
    Resize an image to a specified target size.

    This function resizes an input image to the given dimensions using OpenCV's resize function.
    The interpolation method used is INTER_AREA, which is generally suitable for reducing image size.

    Parameters:
    - image (numpy array): The original image to be resized.
    - target_size (tuple of int, optional): The desired dimensions (width, height) for the resized image. 
                                            Defaults to (256, 256).

    Returns:
    - numpy array: The resized image.
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def crop_image(image, crop_size, step_size):
    """
    Crop an image into smaller segments based on specified size and step.

    This function divides a larger image into smaller segments or "crops." Each crop is defined by 
    crop_size, and the sliding window moves by step_size to create overlapping or non-overlapping crops 
    depending on the step size. The function also records the position (x, y) of each crop relative to 
    the original image.

    Parameters:
    - image (numpy array): The original image to be cropped.
    - crop_size (tuple of int): The dimensions (height, width) of each crop.
    - step_size (int): The sliding window's step size for cropping.

    Returns:
    - list of numpy arrays: The list containing all the cropped images.
    - list of tuples: The list containing the (x, y) positions of each crop.
    """
    cropped_images = []
    positions = []

    # Slide a window of crop_size over the image with a step of step_size
    for y in range(0, image.shape[0] - crop_size[0] + 1, step_size):
        for x in range(0, image.shape[1] - crop_size[1] + 1, step_size):
            # Crop the image and add it to the list
            cropped_img = image[y:y + crop_size[0], x:x + crop_size[1]]
            cropped_images.append(cropped_img)
            positions.append((x, y))

    return cropped_images, positions

def binarize_mask(mask, threshold=0.5):
    """
    Convert a grayscale mask to binary format using a specified threshold.

    This function applies thresholding to a given mask. Each pixel intensity in the mask
    is compared to the threshold value: pixels with intensities higher than the threshold
    are set to 1 (white), and those below or equal to the threshold are set to 0 (black).

    Parameters:
    - mask (numpy array): The input grayscale mask.
    - threshold (float, optional): The threshold value used for binarization. Defaults to 0.5.

    Returns:
    - numpy array: A binary mask with the same dimensions as the input mask.
    """
    return (mask > threshold).astype(np.uint8)

def refine_mask(mask, closing_kernel_size=50, line_threshold=100):
    """
    Enhance and refine a binary mask using morphological operations and contour filtering.

    This function applies morphological closing to the mask to close small gaps and remove noise.
    It then filters out contours that are shorter than a specified threshold, keeping only 
    those that represent significant linear structures, which is particularly useful in road 
    segmentation contexts.

    Parameters:
    - mask (numpy array): The input binary mask.
    - closing_kernel_size (int): Size of the square kernel used for the closing operation.
    - line_threshold (int): Minimum length for a contour to be retained.

    Returns:
    - numpy array: The refined mask.
    """
    # Closing operation to close small holes in the mask
    closing_kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)

    # Contour detection and filtering based on contour length
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(closed_mask)
    for contour in contours:
        if cv2.arcLength(contour, False) > line_threshold:
            cv2.drawContours(filtered_mask, [contour], -1, (255), thickness=cv2.FILLED)

    return filtered_mask

def reassemble_masks(cropped_masks, positions, original_shape, crop_size):
    """
    Reassemble a series of cropped masks into a single mask with the original image's dimensions.

    This function reconstructs the full-size mask from individual cropped masks. It places each
    cropped mask back into its original position, ensuring that the entire area of the original 
    image is covered. Overlapping areas are averaged to blend the masks smoothly.

    Parameters:
    - cropped_masks (list of numpy arrays): List of cropped masks.
    - positions (list of tuples): Positions (x, y) where each cropped mask corresponds in the original image.
    - original_shape (tuple): The shape (height, width) of the original mask.
    - crop_size (tuple): The size of each cropped mask.

    Returns:
    - numpy array: The reassembled mask covering the entire area of the original image.
    """
    full_mask = np.zeros(original_shape[:2])
    count_map = np.zeros(original_shape[:2])

    # Place each cropped mask back into its original position
    for mask, (x, y) in zip(cropped_masks, positions):
        full_mask[y:y+crop_size[0], x:x+crop_size[1]] += mask
        count_map[y:y+crop_size[0], x:x+crop_size[1]] += 1

    # Normalize overlapping areas by averaging
    full_mask /= count_map
    return full_mask


def crop_images_and_masks(images, masks, crop_size=(256, 256), step_size=68):
    """
    Fait glisser une fenêtre de taille crop_size sur les images et les masques 
    et extrait des sous-images et sous-masques correspondants.
    
    :param images: Liste des images.
    :param masks: Liste des masques correspondants.
    :param crop_size: Taille de la fenêtre de découpage (hauteur, largeur).
    :param step_size: Le pas de glissement de la fenêtre.
    :return: Listes des sous-images et sous-masques découpés.
    """
    cropped_images = []
    cropped_masks = []

    for img, mask in zip(images, masks):
        # S'assurer que l'image et le masque ont les mêmes dimensions
        assert img.shape[:2] == mask.shape[:2], "Dimensions de l'image et du masque ne correspondent pas."

        # Glisser la fenêtre sur l'image et le masque
        for y in range(0, img.shape[0] - crop_size[0] + 1, step_size):
            for x in range(0, img.shape[1] - crop_size[1] + 1, step_size):
                cropped_img = img[y:y + crop_size[0], x:x + crop_size[1]]
                cropped_mask = mask[y:y + crop_size[0], x:x + crop_size[1]]

                cropped_images.append(cropped_img)
                cropped_masks.append(cropped_mask)

    return cropped_images, cropped_masks

def load_data(images_path, groundtruth_path, isCropped=False, isResized=True, resize_size=(256, 256), crop_size=(100, 100),step_size=68):
    """
    Load images and their corresponding masks from specified directories, optionally applying resizing and cropping.

    This function loads image and mask pairs, processes them according to the specified parameters, and returns them as numpy arrays. It supports optional resizing and cropping to generate multiple sub-images (crops) from a single image.

    Parameters:
    - images_path (str): Path to the directory containing images.
    - groundtruth_path (str): Path to the directory containing corresponding masks.
    - isCropped (bool, optional): Apply cropping to images and masks if True. Defaults to False.
    - isResized (bool, optional): Resize images and masks if True. Defaults to True.
    - resize_size (tuple, optional): Target size for resizing images and masks. Defaults to (256, 256).
    - crop_size (tuple, optional): Size of each crop for cropping. Defaults to (100, 100).
    - step_size (int, optional): Step size for cropping. Defaults to 68.

    Returns:
    - tuple of numpy arrays: Arrays of images and their corresponding masks.
    """

    images = []
    masks = []
    image_filenames = [f for f in sorted(os.listdir(images_path)) if not f.startswith('.')]
    mask_filenames = [f for f in sorted(os.listdir(groundtruth_path)) if not f.startswith('.')]
    for img_name, mask_name in zip(image_filenames, mask_filenames):
        # Charger l'image
        img_path = os.path.join(images_path, img_name)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img) / 255.0
        # Redimensionner l'image si nécessaire
        if isResized:
            img = resize_image(img, resize_size)
        images.append(img)

        # Charger le masque correspondant
        mask_path = os.path.join(groundtruth_path, mask_name)
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask) / 255.0

        # Redimensionner le masque si nécessaire
        if isResized:
            mask = resize_image(mask, resize_size)
        masks.append(mask)
    if isCropped:
        images, masks = crop_images_and_masks(images, masks, crop_size=crop_size,step_size=step_size)
    
    return np.array(images), np.array(masks)

# Augmentation des données
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)
])

class NumpyDataset(Dataset):
    """
    Custom dataset for loading images and masks, compatible with PyTorch data loaders.

    This dataset class facilitates loading batches of image-mask pairs, optionally applying data augmentation techniques.

    Attributes:
    - images (list): List of images in the dataset.
    - masks (list): List of corresponding masks in the dataset.

    Methods:
    - __len__: Returns the number of items in the dataset.
    - __getitem__: Retrieves an image-mask pair by index, with optional data augmentation.
    """

    def __init__(self, images, masks):
        """
        Initialize the dataset with images, masks, and optional augmentations.

        Parameters:
        - images (list of numpy arrays): List of images in the dataset.
        - masks (list of numpy arrays): List of corresponding masks.
        """
        self.images = images
        self.masks = masks

    def __len__(self):
        """
        Return the number of items in the dataset.

        Returns:
        - int: The number of image-mask pairs in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieve an image-mask pair by index, applying data augmentation as defined in `train_transforms`.

        Parameters:
        - idx (int): Index of the image-mask pair to retrieve.

        Returns:
        - tuple: An image-mask pair as PyTorch tensors.
        """
        image = self.images[idx]
        mask = self.masks[idx].squeeze()

        # Apply data augmentation
        image = Image.fromarray((image * 255).astype(np.uint8))
        image = train_transforms(image)
        image = np.array(image) / 255.0

        # Convert to PyTorch tensors
        image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        mask = torch.from_numpy(mask).float()

        return image, mask



class UNetModel(pl.LightningModule):
    """
    Custom U-Net model class for road segmentation tasks, using PyTorch Lightning.

    Attributes:
    - model (nn.Module): U-Net model with a ResNet50 encoder.
    - criterion (nn.Module): Loss function, Binary Cross-Entropy with Logits.
    - learning_rate (float): Learning rate for the optimizer.

    Methods:
    - forward: Passes input through the model.
    - training_step: Performs a training step and logs metrics.
    - configure_optimizers: Sets up the optimizer.
    """

    def __init__(self, learning_rate=0.0001):
        """
        Initializes the U-Net model with a ResNet50 encoder.

        Parameters:
        - learning_rate (float): Learning rate for the optimizer.
        """
        super(UNetModel, self).__init__()
        self.model = smp.Unet(encoder_name="resnet50", 
                              encoder_weights="imagenet", 
                              in_channels=3, 
                              classes=1)
        self.criterion = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        """
        Forward pass through the model.

        Parameters:
        - x (Tensor): Input tensor.

        Returns:
        - Tensor: Output tensor of the model.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Performs a training step using a batch of data.

        Parameters:
        - batch (tuple): A batch of data (inputs and labels).
        - batch_idx (int): Batch index.

        Returns:
        - Tensor: Computed loss for the batch.
        """
        inputs, labels = batch
        outputs = self(inputs)

        outputs = outputs.squeeze(1)  # Remove channel dimension if necessary
        loss = self.criterion(outputs, labels)

        preds = torch.sigmoid(outputs) > 0.5
        acc = (preds == labels).float().mean()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.

        Returns:
        - Optimizer: The Adam optimizer with the specified learning rate.
        """
        return optim.Adam(self.model.parameters(), lr=self.learning_rate)


def display_mask_grid(cropped_masks, positions, os_dir, title, num_rows=5, num_cols=5):
    """
    Displays a grid of cropped masks for visualization.

    Parameters:
    - cropped_masks (list): List of cropped mask images.
    - positions (list): List of positions where each cropped mask was taken from the original image.
    - os_dir (str): Directory path for saving the plot.
    - title (str): Title for the plot.
    - num_rows (int): Number of rows in the grid.
    - num_cols (int): Number of columns in the grid.
    """
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(cropped_masks):
            ax.imshow(cropped_masks[i], cmap='gray')
            ax.set_title(f"Position: {positions[i]}")
            ax.axis('off')
        else:
            ax.axis('off')

    subtitle = '256x256 masks predictions'
    full_title = os_dir + title + subtitle
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(full_title)
    plt.close()

def save_mask_to_file(output_folder, subdir, title, mask, suffix):
    """
    Saves a mask to a file in the specified output directory.

    Parameters:
    - output_folder (str): Directory to save the output masks.
    - subdir (str): Subdirectory name for organizing mask files.
    - title (str): Base title for the saved mask file.
    - mask (numpy array): The mask to be saved.
    - suffix (str): Suffix to add to the file name for differentiation.

    This function saves the provided mask as a PNG file with a specified name pattern.
    """
    # Construct the file name and path
    file_name = f"{subdir}{title}{suffix}.png"
    file_path = os.path.join(output_folder, file_name)

    # Convert mask to an image format and save
    mask_image = (mask * 255).astype(np.uint8)
    cv2.imwrite(file_path, mask_image)
    print(f"Mask saved to {file_path}")

def run_model_test(model_path, images_folder, output_folder, crop_size, step_size, 
                   isCropped=True, save_masks=False, sample_size_save_masks=10, 
                   binarization_threshold=0.5, plots_dir='masks_predictions', title=''):
    """
    Executes the model on a set of test images, generates predictions, and optionally saves the results.

    Parameters:
    - model_path (str): Path to the trained model file.
    - images_folder (str): Directory containing test images.
    - output_folder (str): Directory to save the output masks.
    - crop_size (tuple): Size of the crops used in the model.
    - step_size (int): Step size for cropping the test images.
    - isCropped (bool): Flag to apply cropping to test images.
    - save_masks (bool): Flag to save the generated masks.
    - sample_size_save_masks (int): Number of sample images for which masks are saved.
    - binarization_threshold (float): Threshold for binarizing the model output.
    - plots_dir (str): Directory for saving plots of mask grids.
    - title (str): Title for the saved plots.

    This function processes each test image, optionally crops it, runs the model to generate masks,
    applies post-processing, and saves the results. It can also generate and save a grid of mask predictions
    for a set of sample images.
    """
    # Load the trained model
    model = UNetModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the test folder
    for subdir in os.listdir(images_folder):
        subdir_path = os.path.join(images_folder, subdir)
        if os.path.isdir(subdir_path):
            image_path = os.path.join(subdir_path, subdir + ".png")
            if os.path.exists(image_path):
                # Load the test image
                test_image = cv2.imread(image_path)
                test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB) / 255.0

                if isCropped:
                    # Crop the image and predict masks
                    cropped_images, positions = crop_image(test_image, crop_size, step_size)
                    cropped_masks = []
                    binarized_cropped_masks = []
                    morpho_cropped_masks = []
                    for img in cropped_images:
                        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float().unsqueeze(0)
                        with torch.no_grad():
                            pred = model(img_tensor)
                        pred_mask = pred.squeeze().cpu().numpy()
                        bin_mask = binarize_mask(pred_mask, threshold=binarization_threshold)
                        binarized_cropped_masks.append(bin_mask)
                        morpho_cropped_masks.append(refine_mask(bin_mask))
                        cropped_masks.append(pred_mask)

                    if sample_size_save_masks > 0:
                        # Display and save grids of masks
                        #print(image_path,sample_size_save_masks)
                        #display_mask_grid(cropped_masks, positions, os_dir=plots_dir, title=title+f'_No_binarize_{sample_size_save_masks}_')
                        #display_mask_grid(binarized_cropped_masks, positions, os_dir=plots_dir, title=title+f'_Binarize_{sample_size_save_masks}_')
                        #display_mask_grid(morpho_cropped_masks, positions, os_dir=plots_dir, title=title+f'_Binarize_and_filters_{sample_size_save_masks}_')
                        sample_size_save_masks -= 1

                    # Reassemble the complete mask
                    full_mask = reassemble_masks(cropped_masks, positions, test_image.shape, crop_size)
                    #full_mask_binarize_before = reassemble_masks(binarized_cropped_masks, positions, test_image.shape, crop_size)
                    full_mask_binarize_after = binarize_mask(full_mask, binarization_threshold)
                    #full_mask_binarize_and_filterized_before = reassemble_masks(morpho_cropped_masks, positions, test_image.shape, crop_size)
                    #full_mask_binarize_and_filterized_after = refine_mask(binarize_mask(full_mask_binarize_after))

                    # Save masks if requested
                    if save_masks:
                        #save_mask_to_file(output_folder, subdir, title, full_mask, '_original')
                        #save_mask_to_file(output_folder, subdir, title, full_mask_binarize_before, '_binarize_b')
                        save_mask_to_file(output_folder, subdir, title, full_mask_binarize_after, '_prediction')
                        #save_mask_to_file(output_folder, subdir, title, full_mask_binarize_and_filterized_before, '_binarizeandfiltered_b')
                        #save_mask_to_file(output_folder, subdir, title, full_mask_binarize_and_filterized_after, '_binarize_andfiltered_a')
                else:
                    # Process without cropping
                    img = resize_image(test_image, (256, 256))
                    img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float().unsqueeze(0)
                    with torch.no_grad():
                        pred = model(img_tensor)
                    pred_mask = pred.squeeze().cpu().numpy()
                    pred_mask_binarize = binarize_mask(pred_mask*255)
                    
                    # Save masks if requested
                    if save_masks:
                        save_mask_to_file(output_folder, subdir, title, pred_mask, '_original')
                        save_mask_to_file(output_folder, subdir, title, pred_mask_binarize, '_binarize')
                        print(f"Mask saved for {subdir}")


def train_and_save_model(images_path, groundtruth_path, crop_size=(256, 256), batch_size=64, isCropped=False, isResized=False, max_epochs=100, patience=7, step_size=68):
    """
    Train the provided model and save the trained model to disk.

    This function handles the entire training pipeline, including data loading, training loop setup, and saving the trained model.

    Parameters:
    - images_path (str): Path to the directory containing the training images.
    - groundtruth_path (str): Path to the directory containing the ground truth masks.
    - crop_size (tuple of int, optional): The size to crop the images to, if cropping is enabled. Defaults to (256, 256).
    - batch_size (int, optional): Number of samples per batch. Defaults to 64.
    - isCropped (bool, optional): Whether to apply cropping to the images. Defaults to False.
    - isResized (bool, optional): Whether to resize the images. Defaults to False.
    - max_epochs (int, optional): Maximum number of training epochs. Defaults to 100.
    - patience (int, optional): Patience for early stopping. Training stops after this many epochs without improvement. Defaults to 7.
    - step_size (int, optional): Step size for cropping. Defaults to 68.

    The function sets up the data loaders, defines callbacks for monitoring training, and initiates the training process using PyTorch Lightning. After training, it saves the trained model state to a specified file.
    """
    model = UNetModel()
    # Load data and create a DataLoader
    images, masks = load_data(images_path, groundtruth_path, crop_size=crop_size, isCropped=isCropped, isResized=isResized, step_size=step_size)
    dataloader = DataLoader(NumpyDataset(images, masks), batch_size=batch_size, shuffle=True)

    # Define save directory and create if not exists
    save_dir = os.path.join(os.path.expanduser('~'), 'models_pytorch')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define callbacks for saving best model and early stopping
    checkpoint_callback = ModelCheckpoint(monitor='train_loss', dirpath=save_dir, filename='unet-{epoch:02d}-{loss:.2f}', save_top_k=1, mode='min')
    early_stop_callback = EarlyStopping(monitor='train_loss', patience=patience, verbose=True, mode='min')

    # Train the model
    trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model, dataloader)

    # Save the model manually after training
    model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# Update these paths according to your directory structure
images_path = '/path/to/training_images'# <===== configure 
groundtruth_path = '/path/to/training_groundtruth'# <===== configure 


train_and_save_model(images_path,groundtruth_path,
                     isCropped=True,
                     crop_size=(256,256),
                     batch_size=128,
                     isResized=False,
                     max_epochs=30,
                     patience=10,
                     step_size=144)

run_model_test(model_path="/path/to/models_pytorch/model.pth", # <===== configure 
               images_folder= "/path/to/test_set_images", # <===== configure 
               output_folder= "/path/to/output_model_images", # <===== configure 
               crop_size=(256, 256),
               step_size=88,
               title = '',
               isCropped=True,
               save_masks=True,
               plots_dir='/path/to/masks_predictions/',# <===== configure 
               binarization_threshold=0.3
               )


### Create the csv file
foreground_threshold = 0.25 

def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))

def masks_to_submission(submission_filename, image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))

if __name__ == '__main__':
    submission_filename = 'submission.csv'
    image_filenames = []
    mask_folder = '/path/to/predictions_masks' # <===== configure 
    for filename in os.listdir(mask_folder):
        if filename.endswith('.png'):
            image_filenames.append(os.path.join(mask_folder, filename))
    masks_to_submission(submission_filename, image_filenames)
