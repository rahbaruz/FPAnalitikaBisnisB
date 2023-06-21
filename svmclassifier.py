import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# Membaca data dari file CSV
data = pd.read_csv("dataset-Jenis-Sampah.csv")

# Memisahkan fitur dan label
X = data.iloc[:, 1:100].values  # Mengambil kolom kedua sampai terakhir sebagai fitur
y = data.iloc[:, 101].values  # Mengambil kolom pertama sebagai label

# Mengubah label menjadi nilai numerik
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Memisahkan data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Address class imbalance
oversampler = RandomOverSampler(random_state=42)
X_train, y_train = oversampler.fit_resample(X_train, y_train)

# Membuat objek Support Vector Classifier
model = SVC(random_state=42)

# Melatih model menggunakan data latih
model.fit(X_train, y_train)

# Memprediksi label untuk data uji
y_pred = model.predict(X_test)

# Membuat confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_mat)

# Menghitung akurasi prediksi
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi:", accuracy)
