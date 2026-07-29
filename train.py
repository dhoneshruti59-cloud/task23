import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
df = pd.read_csv("music_streaming_habits_2026.csv")

# Features and Target
X = df.drop(["listener_id", "subscription"], axis=1)
y = df["subscription"]

# One Hot Encoding
X = pd.get_dummies(X, drop_first=True)

# Save column names
columns = X.columns.tolist()

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- Logistic Regression ----------------
logistic = LogisticRegression(max_iter=1000)
logistic.fit(X_train, y_train)

pred = logistic.predict(X_test)

print("\n===== Logistic Regression =====")
print("Accuracy:", accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(logistic, "logistic.pkl")

# ---------------- Decision Tree ----------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

pred = dt.predict(X_test)

print("\n===== Decision Tree =====")
print("Accuracy:", accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(dt, "decision_tree.pkl")

# ---------------- SVM ----------------
svm = SVC()
svm.fit(X_train, y_train)

pred = svm.predict(X_test)

print("\n===== SVM =====")
print("Accuracy:", accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(svm, "svm.pkl")

# ---------------- KNN ----------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)

print("\n===== KNN =====")
print("Accuracy:", accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(knn, "knn.pkl")

# ---------------- Naive Bayes ----------------
nb = GaussianNB()
nb.fit(X_train, y_train)

pred = nb.predict(X_test)

print("\n===== Naive Bayes =====")
print("Accuracy:", accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(nb, "naive_bayes.pkl")

# Save Scaler and Columns
joblib.dump(scaler, "scaler.pkl")
joblib.dump(columns, "columns.pkl")

print("\nAll models trained successfully.")
print("PKL files saved successfully.")