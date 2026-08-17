from sklearn.tree import DecisionTreeClassifier


def build_model():
    """Create and return the Decision Tree model."""
    return DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )