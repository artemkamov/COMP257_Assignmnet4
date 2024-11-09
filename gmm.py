#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.ndimage import rotate


# 1. Retrieve and load the Olivetti faces dataset
faces_data = fetch_olivetti_faces(shuffle=True, random_state=42)

# X contains the flattened image data, y contains the labels
X = faces_data.data
y = faces_data.target

# 2. Split the Dataset using Stratified Sampling
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Verify stratified sampling by checking the distribution of images per person
unique_train, counts_train = np.unique(y_train, return_counts=True)
unique_val, counts_val = np.unique(y_val, return_counts=True)
unique_test, counts_test = np.unique(y_test, return_counts=True)

print("\nTrain set counts per person:", dict(zip(unique_train, counts_train)))
print("\nValidation set counts per person:", dict(zip(unique_val, counts_val)))
print("\nTest set counts per person:", dict(zip(unique_test, counts_test)))


# Step 3: Apply PCA, preserving 99% of variance
pca = PCA(n_components=0.99, svd_solver='full')
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

print(f"\nOriginal shape of X_train: {X_train.shape}")
print(f"Reduced shape of X_train after PCA: {X_train_pca.shape}")


# 4. Determine the most suitable covariance type for the dataset
# Test different covariance types and compute AIC/BIC
covariance_types = ['spherical', 'tied', 'diag', 'full']
aic_scores = []
bic_scores = []

for cov_type in covariance_types:
    gmm = GaussianMixture(n_components=10, covariance_type=cov_type, random_state=42)
    gmm.fit(X_train_pca)
    aic_scores.append(gmm.aic(X_val_pca))
    bic_scores.append(gmm.bic(X_val_pca))

# Display AIC and BIC scores for each covariance type
# Format and print AIC and BIC scores in a more readable way
print("\nAIC scores for covariance types:")
for cov_type, score in zip(covariance_types, aic_scores):
    print(f"{cov_type.capitalize()}: {score:.2f}")

print("\nBIC scores for covariance types:")
for cov_type, score in zip(covariance_types, bic_scores):
    print(f"{cov_type.capitalize()}: {score:.2f}")

# Choose the covariance type with the lowest BIC
best_cov_type = covariance_types[np.argmin(bic_scores)]
print(f"\nBest covariance type based on BIC: {best_cov_type}")


# 5. Determine the optimal number of clusters with BIC
min_clusters = 2
max_clusters = 20
bic = []

for n in range(min_clusters, max_clusters + 1):
    gmm = GaussianMixture(n_components=n, covariance_type=best_cov_type, random_state=42)
    gmm.fit(X_train_pca)
    bic.append(gmm.bic(X_val_pca))

# Find the optimal number of clusters
optimal_clusters = np.argmin(bic) + min_clusters
print(f"\nOptimal number of clusters based on BIC: {optimal_clusters}")


# 6. Plot the results from steps 3 and 4
# Plot 1: Explained variance by PCA components
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of PCA Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of PCA Components')
plt.grid(True)
plt.show()

# Plot 2: AIC scores for different covariance types
plt.figure(figsize=(10, 6))
plt.bar(covariance_types, aic_scores, color='skyblue')
plt.xlabel('Covariance Type')
plt.ylabel('AIC Score')
plt.title('AIC Scores for Different Covariance Types')
plt.grid(True)
plt.show()

# Plot 3: BIC scores for different covariance types
plt.figure(figsize=(10, 6))
plt.bar(covariance_types, bic_scores, color='skyblue')
plt.xlabel('Covariance Type')
plt.ylabel('BIC Score')
plt.title('BIC Scores for Different Covariance Types')
plt.grid(True)
plt.show()

# Plot 4: BIC scores vs. number of clusters
plt.figure(figsize=(10, 6))
plt.plot(range(min_clusters, max_clusters + 1), bic, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('BIC Score')
plt.title('BIC Score vs. Number of Clusters')
plt.grid(True)
plt.show()


# 7. Output the hard clustering assignments for each instance
# Fit GMM with the optimal number of clusters and best covariance type
gmm = GaussianMixture(n_components=optimal_clusters, covariance_type=best_cov_type, random_state=42)
gmm.fit(X_train_pca)

# Hard clustering assignments (cluster labels for each instance)
hard_assignments = gmm.predict(X_train_pca)

# Summary of hard clustering assignments
unique, counts = np.unique(hard_assignments, return_counts=True)
hard_assignment_summary = dict(zip(unique, counts))
print("\nHard Clustering Assignments Summary:", hard_assignment_summary)

# Display the first 10 instances' hard clustering assignments
print("\nHard Clustering Assignments (first 10 instances):")
for i, assignment in enumerate(hard_assignments[:10], start=1):
    print(f"Instance {i}: Cluster {assignment}")
    
# Visualize hard clustering images for the first 10 instances
plt.figure(figsize=(15, 7))
for i in range(10):
    plt.subplot(2, 5, i + 1)  # Arrange in a 2x5 grid for better viewing
    plt.imshow(X_train[i].reshape(64, 64), cmap='gray')
    plt.title(f"Hard Cluster {hard_assignments[i]}")
    plt.axis('off')
plt.suptitle("First 10 Images - Hard Clustering")
plt.show()

# 8. Output the soft clustering probabilities for each instance
soft_assignments = gmm.predict_proba(X_train_pca)

# Display the first 10 instances' soft clustering probabilities
print("\nSoft Clustering Probabilities (first 10 instances):")
for i, probs in enumerate(soft_assignments[:10], start=1):
    formatted_probs = [f"{p:.2f}" for p in probs]
    print(f"Instance {i}: " + ", ".join([f"Cluster {j}: {formatted_probs[j]}" for j in range(len(probs))]))


# Visualize soft clustering images for the first 10 instances
plt.figure(figsize=(15, 7))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i].reshape(64, 64), cmap='gray')
    plt.title(f"Soft Cluster 0: {soft_assignments[i][0]:.2f}")  
    plt.axis('off')
plt.suptitle("First 10 Images - Soft Clustering")
plt.show()


# 9. Generate and visualize new faces
# Generate new samples
generated_faces_pca, _ = gmm.sample(20)  # Generate 5 new samples in the PCA space

# Transform back to the original space
generated_faces = pca.inverse_transform(generated_faces_pca)

# Plot generated faces
plt.figure(figsize=(15, 10))  # Adjust the figure size as needed
for i in range(20):
    plt.subplot(4, 5, i + 1)  # 4 rows and 5 columns
    plt.imshow(generated_faces[i].reshape(64, 64), cmap='gray')
    plt.axis('off')
plt.suptitle("Generated Faces")
plt.tight_layout()
plt.show()


# 10. Modify some images
num_images_to_modify = 20
original_images = X_train[:num_images_to_modify]

# Apply transformations
modified_images = []
for img in original_images:
    img_reshaped = img.reshape(64, 64)
    # Rotate by 180 degrees
    rotated_img = rotate(img_reshaped, angle=180, reshape=False)
    # Flip horizontally
    flipped_img = np.fliplr(img_reshaped)
    # Darken the image by reducing pixel intensity
    darkened_img = np.clip(img_reshaped - 0.5, 0, 255)

    # Flatten the modified images back to the original shape
    modified_images.extend([rotated_img.flatten(), flipped_img.flatten(), darkened_img.flatten()])

# Visualize modified images
plt.figure(figsize=(12, 5))
for i, modified_img in enumerate(modified_images[:3]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(modified_img.reshape(64, 64), cmap='gray')
    plt.axis('off')
plt.suptitle("Examples of Modified (Anomalous) Faces")
plt.show()

# 11. Calculate log likelihoods for normal and modified images
normal_scores = gmm.score_samples(X_train_pca[:num_images_to_modify])
modified_pca = pca.transform(modified_images)  # Transform modified images to PCA space
anomaly_scores = gmm.score_samples(modified_pca)

# Display log likelihoods for normal images with numbering
print("\nLog Likelihoods for Normal Images:")
for i, score in enumerate(normal_scores, start=1):
    print(f"Image {i}: {score:.2f}")

# Display log likelihoods for modified images with transformations labeled
print("\nLog Likelihoods for Modified (Anomalous) Images:")
for i in range(num_images_to_modify):
    print(f"Image {i+1} rotated: {anomaly_scores[i * 3]:.2f}")
    print(f"Image {i+1} flipped: {anomaly_scores[i * 3 + 1]:.2f}")
    print(f"Image {i+1} darkened: {anomaly_scores[i * 3 + 2]:.2f}")
