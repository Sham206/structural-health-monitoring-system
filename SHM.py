import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Generate synthetic sensor data
np.random.seed(42)
num_samples = 1000

data = {
    'vibration': np.random.normal(0.5, 0.1, num_samples),
    'strain': np.random.normal(0.3, 0.05, num_samples),
    'displacement': np.random.normal(0.2, 0.04, num_samples),
    'temperature': np.random.normal(25, 2, num_samples),
    'humidity': np.random.normal(60, 5, num_samples),
}

# Damage simulation logic based on sensor thresholds
def simulate_damage(vib, strain, disp, temp, hum):
    risk = 0.6 * vib + 0.4 * strain + 0.3 * disp
    if temp > 30 or hum > 70:
        risk += 0.2
    return 1 if risk > 0.8 else 0

# Assign status based on simulated function
status = [
    simulate_damage(data['vibration'][i], data['strain'][i], data['displacement'][i],
                    data['temperature'][i], data['humidity'][i])
    for i in range(num_samples)
]
data['status'] = status

# Create DataFrame and save it
df = pd.DataFrame(data)
df.to_csv("SHM_sensor_data.csv", index=False)

# 2. Train ML model
X = df.drop('status', axis=1)
y = df['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 3. Evaluation
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("SHM_model_report.csv")  # Save classification report

# 4. Feature importance
feature_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
feature_imp.plot(kind='bar')
plt.title('Feature Importance')
plt.xlabel("Features")
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig("SHM_feature_importance.png")

# 5. Real-time simulation
alerts = []
for i in range(20):
    test_input = np.random.normal(loc=[0.5, 0.3, 0.2, 25, 60], scale=[0.1, 0.05, 0.04, 2, 5])
    test_df = pd.DataFrame([test_input], columns=X.columns)
    pred = clf.predict(test_df)[0]
    if pred == 1:
        alerts.append(f"ALERT {i+1}: Damage predicted for sensor reading {test_input.tolist()}")

# Save alerts to a text file
with open("SHM_alerts.txt", 'w') as f:
    for line in alerts:
        f.write(line + '\n')
