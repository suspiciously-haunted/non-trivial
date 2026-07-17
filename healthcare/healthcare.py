import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('healthcare_dataset.csv')
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)

# train_df.fillna(method='ffill', inplace=True)
# test_df.fillna(method='ffill', inplace=True)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_df["Medical Condition"])

X = pd.get_dummies(train_df.drop(["Medical Condition", "Name"], axis=1))
feature_names = X.columns

X_test = pd.get_dummies(test_df.drop("Name", axis=1))
X_test = X_test.reindex(columns=feature_names, fill_value=0)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Validation Accuracy: {acc:.2f}")

test_preds = best_model.predict(X_test_scaled)
decoded_preds = label_encoder.inverse_transform(test_preds)

results = pd.DataFrame({
    "Name": test_df["Name"],
    "Predicted Condition": decoded_preds
})

print(results)
