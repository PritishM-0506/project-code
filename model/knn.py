from sklearn.neighbors import KNeighborsClassifier


def build_model():
    """Create and return the K-Nearest Neighbors model."""
    return KNeighborsClassifier(
        n_neighbors=5
    )