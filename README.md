# Machine-Learning-Notes

KNN Model

# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# Create feature and target arrays
X = np.array(digits.data)
y = np.array(digits.target)
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)
# Create a k-NN classifier with 7 neighbors: knn
knn = knn = KNeighborsClassifier(n_neighbors=7)
# Fit the classifier to the training data
knn.fit(X, y)

Output = Accuracy
