# Import Library
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Dataset
df = pd.read_csv('kf_analytics_data.csv')
print(df)

## Information of Dataset
print(df.info())
print(df.describe())

# Prepocessing Data
## Checking Missing Value
print(df.isna().sum())

# Visualisasi heatmap missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cmap='viridis', cbar=False)
plt.title('Missing Values Heatmap')
plt.show()

### Tidak ada missing value

# Analisis Data
df_final = df[['rating_transaksi', 'actual_price', 'discount_percentage', 'rating_cabang']]
# Mengubah isi kolom dari koma menjadi titik
# Mengubah isi kolom dari koma menjadi titik
df_final['rating_transaksi'] = df_final['rating_transaksi'].str.replace(',', '.')
#df_final['actual_price'] = df_final['actual_price'].str.replace(',', '.')
df_final['discount_percentage'] = df_final['discount_percentage'].str.replace(',', '.')
df_final['rating_cabang'] = df_final['rating_cabang'].str.replace(',', '.')

# Mengubah tipe data kolom menjadi float
df_final['rating_transaksi'] = df_final['rating_transaksi'].astype(float)
#df_final['actual_price'] = df_final['actual_price'].astype(float)
df_final['discount_percentage'] = df_final['discount_percentage'].astype(float)
df_final['rating_cabang'] = df_final['rating_cabang'].astype(float)

# Menampilkan dataframe setelah perubahan
print(df_final)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Memisahkan fitur (features) dan target (target) dari dataframe
X = df_final.drop(['actual_price'], axis=1)
y = df_final['actual_price']  # Ganti 'target_column' dengan nama kolom target Anda

# Memisahkan data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Melatih model menggunakan data latih
model.fit(X_train, y_train)

# Memprediksi kelas target menggunakan data uji
y_pred = model.predict(X_test)

# Menghitung akurasi model
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi model Random Forest:", accuracy)
