def split_dataset(X, y):
    # X_train, X_test, y_train, y_test
    return  X[:60000], X[60000:], y[:60000], y[60000:]