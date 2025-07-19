from src.train import load_config, train_model
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def test_load_config():
    config = load_config("config/config.json")
    assert 'C' in config and isinstance(config['C'], float)
    assert 'solver' in config and isinstance(config['solver'], str)
    assert 'max_iter' in config and isinstance(config['max_iter'], int)

def test_train_model_object():
    digits = load_digits()
    X, y = digits.data, digits.target
    config = load_config("config/config.json")
    model = train_model(X, y, config)
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")

def test_train_model_accuracy():
    digits = load_digits()
    X, y = digits.data, digits.target
    config = load_config("config/config.json")
    model = train_model(X, y, config)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    assert acc > 0.85
