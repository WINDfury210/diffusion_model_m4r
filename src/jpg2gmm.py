import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.mixture import GaussianMixture
from PIL import Image


def fit_image(directory, n_components=5, store_path=False):
    """
    Fit a Gaussian Mixture Model (GMM) using images in a directory as sample points.
    
    Parameters:
    - directory: Path to the directory containing image files.
    - n_components: Number of Gaussian components for the GMM.
    
    Returns:
    - gmm: Fitted Gaussian Mixture Model for the image samples.
    - images_array: 2D array of flattened image vectors.
    """
    images = []
    # Iterate through all .jpg files in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(directory, filename)
            
            # Load the image using PIL
            image = Image.open(image_path)
            image = np.array(image) / 255.0  # Normalize to [0, 1]
            
            # Flatten the image into a vector (reshape)
            flattened_image = image.flatten()
            images.append(flattened_image)
    
    if len(images) == 0:
        raise ValueError("No .jpg images found in the directory.")
    
    # Convert list of image vectors into a 2D array (shape: [num_images, num_features])
    images_array = np.vstack(images)
    
    # Fit the GMM using the flattened image vectors
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(images_array)
    if store_path:
        store_gmm_model(gmm, f"/Users/apple/Desktop/IC/700/M4R/codes/models/{store_path}.joblib")
    else:
        return gmm


def display_image(image, title="Image"):
    """
    Display a single image, whether from a file path or a NumPy array.

    Parameters:
    - image: Path to the image file (str) or a NumPy array representing the image.
    - title: Title for the displayed image.
    """
    # Check if the input is a file path or a NumPy array
    if isinstance(image, str):  # If it's a string, treat it as a file path
        img_to_show = Image.open(image)
    elif isinstance(image, np.ndarray):  # If it's a NumPy array, convert to PIL Image
        img_to_show = Image.fromarray((image * 255).astype(np.uint8))
    else:
        raise ValueError("Input must be either a file path (str) or a NumPy array.")

    # Plot the image
    plt.figure(figsize=(5, 5))
    plt.imshow(img_to_show)
    plt.title(title)
    plt.axis('off')
    plt.show()
    

def store_gmm_model(gmm, model_filename):
    """
    Store the fitted GMM model to a file.

    Parameters:
    - gmm: The fitted GaussianMixture model.
    - model_filename: The filename to save the model to.
    """
    joblib.dump(gmm, model_filename)
    print(f"GMM model saved as {model_filename}")


def load_gmm_model(model_filename):
    """
    Load a pre-trained GMM model from a file.

    Parameters:
    - model_filename: The filename of the saved GMM model.

    Returns:
    - gmm: The loaded GaussianMixture model.
    """
    gmm = joblib.load(model_filename)
    print(f"GMM model loaded from {model_filename}")
    return gmm



