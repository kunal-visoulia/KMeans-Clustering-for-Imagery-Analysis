# KMeans-Clustering-for-Imagery-Analysis
Use a K-means algorithm to perform image classification
>**Through deep learning 99%+ accuracy acchieved but with kmeans that much is not achievable. My Aim was to understand kmeans clustering algorithm and high dimensional data.In deep neural n/w's, you can't visualize cluster centroids so that's why I used kmeans**

## DATASET and PREPROCESSING
The MNIST database of handwritten digits, has a training set of 60,000 examples, and a test set of 10,000 examples. The digits have been size-normalized and centered in a fixed-size image.

**Preprocess images for clustering: converting 2D image array(array of pixel intensities:28x28) to 1D array(784x1) for ingestion by Kmeans algorithm**

The data set had 0-9 digit images, example dataset:<br/>
![](images/1.png)

- Assuming 10 be the number of clusters(for 10 digits), I fit the MiniBatchKmeans algorithm(sue to large dataset, 60K x 784).
- Also KMeans being an unsupervised learning algorithm,it assigned clusters to dataset. Then I mapped each cluster number to appropriate integer number(the most common integer in the cluster because the number of cluster was just 10 and there would obviously be some misclassfications)
- Optimizing and Evaluating the Clustering Algorithm: Varied number of clusters as follows:10,16,36,64,144,256,400 and compared Inertia, Homogeneity, Accuracy. **Achieved an accuracy of 99.3 with 256 clusters adn 90% with 400 clusters but 400 clusters took much more computation time with not much increase in accuracy, so chose 256 clusters to be optimal(tradeoff)**
- Visualizing Cluster Centroids: The most representative point within each cluster is called the centroid. ***For visualization purpose, I set the number of cluster to 36***

![](images/1.png)

