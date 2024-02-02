import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import time

# Load the data set (You must specify the path to your TUANDROMD.csv file) / Veri setini yükleyin (TUANDROMD.csv dosyanızın yolunu belirtmelisiniz)
dataset = pd.read_csv('TUANDROMD.csv')

# Separate independent variables and target variable / Yeniden bağımsız değişkenleri ve hedef değişkenini ayırın
X = dataset.drop("Label", axis=1)
y = dataset["Label"]

# Create training and test data sets / Eğitim ve test veri setlerini oluşturun
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200)

# Create the Random Forest Model / Random Forest modelini oluşturun
clf = DecisionTreeClassifier()(n_estimators=100, random_state=42)

# Measure training time / Eğitim süresini ölçün 
start_time = time.time()
clf.fit(X_train, y_train)
training_time = time.time() - start_time
print("Traning Time:", training_time)

# Measure test time / Test süresini ölçün 
start_time = time.time()
y_pred = clf.predict(X_test)
testing_time = time.time() - start_time
print("Testing Time:", testing_time)

#  Measure Precision, Recall, Accuracy, F1 Score / Precision, Recall, Accuracy, F1 Score hesaplayın 
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
accuracy = metrics.accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# Print values / Değerleri yazdırın  
print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("\n")

# Print Confusion Matrix / Confusion Matrix'i yazdırın 
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)