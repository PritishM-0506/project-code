from sklearn.linear_model import LogisticRegression


def build_model():
    """Create and return the Logistic Regression model."""
    return LogisticRegression(
        max_iter=2000,
        random_state=42
    )