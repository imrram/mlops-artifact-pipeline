import joblib
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score

digits = load_digits()
X, y = digits.data, digits.target

model = joblib.load("model_train.pkl")
y_pred = model.predict(X)

acc = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred, average='weighted')

print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
