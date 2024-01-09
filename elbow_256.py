import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from PIL import UnidentifiedImageError  # Add this import statement

weights_url = 'http://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
weights_path = tf.keras.utils.get_file('vgg16_weights.h5', weights_url, cache_subdir='models')

# Step 1: Choose a Pre-trained Model
base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))

# Load the weights from the local file
base_model.load_weights(weights_path)

# Specify the root directory containing images
root_dir = '/Users/anh/Desktop/last_lab/FinalProject/truth_label_0.15_remove_empty_cluster'

# Step 2: Iterate through all images in the directory and its subdirectories
features_list = []

for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        img_path = os.path.join(subdir, file)
        
        try:
            # Process the image
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            # Extract features using the pre-trained model
            features = base_model.predict(x)
            features = features.flatten()  # Flatten the feature vector
            features_list.append(features)
        except (UnidentifiedImageError, OSError):
            # Skip files that cannot be identified as images
            print(f"Skipping {img_path}")

# Convert the features list to a numpy array
X = np.array(features_list)

# Step 3: Apply PCA
pca = PCA(n_components=10)  # You can adjust the number of components based on your dataset
X_pca = pca.fit_transform(X)

# Step 4: Elbow Method to Determine Optimal Number of Clusters
wcss = []  # Within-Cluster Sum of Squares

for i in range(1, 100):  # Try different numbers of clusters
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)

# Plot the Elbow
plt.plot(range(1, 100), wcss, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.show()

# Rest of the code (choose optimal number of clusters, apply K-Means clustering, create folders, copy images, visualize clusters) remains the same...
