import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import shutil
from sklearn.metrics import pairwise_distances_argmin_min


weights_url = 'http://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
weights_path = tf.keras.utils.get_file('vgg16_weights.h5', weights_url, cache_subdir='models')

# Step 1: Choose a Pre-trained Model
base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))

# Load the weights from the local file
base_model.load_weights(weights_path)

dataset_path = '/Users/anh/Desktop/last_lab/FinalProject/mix'
image_paths = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path)]

features_list = []
image_names = []

for img_path in image_paths:
    img_name = os.path.basename(img_path)
    image_names.append(img_name)

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    features = base_model.predict(x)
    features = features.flatten()  # Flatten the feature vector
    features_list.append(features)

X = np.array(features_list)

# Step 3: Reduce Dimensionality with PCA (optional but recommended)
pca = PCA(n_components=10)  # You can adjust the number of components based on your dataset
X_pca = pca.fit_transform(X)

# Step 4: Apply K-Means Clustering
num_clusters = 400  # You can adjust the number of clusters based on your dataset
kmeans = KMeans(n_clusters=num_clusters, max_iter=300, random_state=42)
cluster_labels = kmeans.fit_predict(X_pca)
centroid_points = kmeans.cluster_centers_  

# Get distances to cluster centers for each sample
distances_to_centers = kmeans.transform(X_pca)

# Calculate the mean distance for each cluster
mean_distances_per_cluster = np.mean(distances_to_centers, axis=0)

# Optionally, print or analyze the mean distances for each cluster
for i, mean_distance_cluster in enumerate(mean_distances_per_cluster):
    print(f"Cluster {i + 1}: Mean Distance to Cluster Center = {mean_distance_cluster}")



# Step 5: Create "truth_label" folder if not exist
output_folder = 'truth_label'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# # Save images to subfolders based on cluster labels
for i in range(num_clusters):
    cluster_folder = os.path.join(output_folder, f'Cluster_{i + 1}')
    if not os.path.exists(cluster_folder):
        os.makedirs(cluster_folder)

for idx, img_path in enumerate(image_paths):
    cluster_label = cluster_labels[idx]
    distance_to_center = distances_to_centers[idx, cluster_label]
    mean_distance_cluster = mean_distances_per_cluster[cluster_label]

    # Only copy if the distance is less than the mean distance for the cluster
    if distance_to_center < mean_distance_cluster * 0.15:
        cluster_folder = os.path.join(output_folder, f'Cluster_{cluster_label + 1}')
        output_path = os.path.join(cluster_folder, os.path.basename(img_path))
        shutil.copy(img_path, output_path)

