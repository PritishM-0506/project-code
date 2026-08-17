from sklearn.ensemble import RandomForestClassifier


def build_model():
    """Create and return the Random Forest model."""
    return RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )