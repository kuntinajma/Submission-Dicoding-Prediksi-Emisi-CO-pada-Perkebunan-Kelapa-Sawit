# -*- coding: utf-8 -*-
"""
Skrip Gabungan untuk Model Prediksi CO2 (Revisi Lanjutan)

Skrip ini mengintegrasikan seluruh alur kerja mulai dari pemuatan dan penggabungan data,
melalui analisis data eksploratif (EDA) dan pra-pemrosesan, hingga pelatihan
model machine learning, evaluasi, dan penyimpanan.

Logika sampling data CO2, urutan kolom, dan normalisasi telah disesuaikan 
agar identik dengan versi notebook untuk memastikan hasil yang konsisten.
Visualisasi telah dikomentari untuk mempercepat pengujian.
"""

# ==============================================================================
# 1. SETUP: IMPORT PUSTAKA
# ==============================================================================
# Pustaka umum untuk manipulasi data dan sistem
import os
import pandas as pd
import numpy as np
import glob
# 'statistics.mode' tidak lagi digunakan untuk menjaga konsistensi dengan notebook asli

# Pustaka untuk visualisasi
import matplotlib.pyplot as plt
import seaborn as sns

# Pustaka untuk Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

print("Pustaka berhasil diimpor.")


# ==============================================================================
# 2. PEMUATAN DATA, SAMPLING, DAN PENGGABUNGAN
# ==============================================================================
print("\n--- Memulai: 2. Pemuatan Data, Sampling, dan Penggabungan ---")

# --- Langkah 2.1: Proses Data CO2 ---
print("Memproses data CO2...")
co2_folder_path = 'dataset/data-raw/co2'
# Membuat direktori jika belum ada (untuk environment di mana direktori mungkin tidak ada)
os.makedirs(co2_folder_path, exist_ok=True)
# Menambahkan file CSV dummy jika direktori kosong untuk menghindari error saat pengujian
if not glob.glob(os.path.join(co2_folder_path, "*.csv")):
    dummy_co2_data = {'timestamp': pd.to_datetime(['2025-04-24 00:00:00', '2025-04-24 00:00:30']), 'co2': [400, 401]}
    dummy_co2_df = pd.DataFrame(dummy_co2_data)
    dummy_co2_df.to_csv(os.path.join(co2_folder_path, 'dummy_co2.csv'), index=False)
    print(f"File CSV dummy 'dummy_co2.csv' dibuat di {co2_folder_path}")

all_co2_files = glob.glob(os.path.join(co2_folder_path, "*.csv"))
if all_co2_files:
    df_co2_list = []
    for f in all_co2_files:
        try:
            df_temp = pd.read_csv(f)
            if 'timestamp' in df_temp.columns and 'co2' in df_temp.columns:
                 df_co2_list.append(df_temp[['timestamp', 'co2']])
            else:
                print(f"Peringatan: File {f} tidak memiliki kolom 'timestamp' atau 'co2'. Dilewati.")
        except Exception as e:
            print(f"Error membaca file {f}: {e}. Dilewati.")

    if df_co2_list:
        df_co2 = pd.concat(df_co2_list, ignore_index=True)
        df_co2['timestamp'] = pd.to_datetime(df_co2['timestamp'], errors='coerce')
        df_co2['co2'] = pd.to_numeric(df_co2['co2'], errors='coerce')
        df_co2.dropna(subset=['timestamp', 'co2'], inplace=True) # Hapus baris jika timestamp atau co2 NaN setelah konversi

        if not df_co2.empty:
            df_co2['minute'] = df_co2['timestamp'].dt.floor('min')
            
            # Menggunakan metode .mode().iloc[0] dari Pandas agar sama persis dengan notebook asli.
            minute_co2 = df_co2.groupby('minute')['co2'].agg(
                lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
            ).reset_index()

            if not minute_co2.empty and 'minute' in minute_co2.columns and not minute_co2['minute'].isnull().all():
                start_minute_co2 = minute_co2['minute'].min()
                end_minute_co2 = minute_co2['minute'].max()
                if pd.isna(start_minute_co2) or pd.isna(end_minute_co2):
                    print("Peringatan: Tidak dapat menentukan rentang waktu untuk data CO2 karena nilai min/max adalah NaT.")
                    start_minute_co2 = pd.to_datetime('2025-04-24 00:00:00')
                    end_minute_co2 = pd.to_datetime('2025-05-07 23:59:00')

                full_minutes_range = pd.date_range(
                    start=start_minute_co2.floor('D'),
                    end=end_minute_co2.ceil('D') - pd.Timedelta(minutes=1),
                    freq='T' # 'T' adalah alias untuk 'min'
                )
                full_minutes_df = pd.DataFrame({'minute': full_minutes_range})
                co2_sampled = pd.merge(full_minutes_df, minute_co2, on='minute', how='left')
                print("Data CO2 di-sample per menit menggunakan metode Pandas (sesuai notebook).")
            else:
                print("Peringatan: Tidak ada data CO2 yang valid untuk di-sample setelah grouping.")
                co2_sampled = pd.DataFrame(columns=['minute', 'co2'])
        else:
            print("Peringatan: DataFrame CO2 kosong setelah pembersihan awal.")
            co2_sampled = pd.DataFrame(columns=['minute', 'co2'])
    else:
        print("Peringatan: Tidak ada file CO2 yang valid yang dapat dibaca.")
        co2_sampled = pd.DataFrame(columns=['minute', 'co2'])
else:
    print(f"Peringatan: Tidak ada file CSV yang ditemukan di {co2_folder_path}.")
    co2_sampled = pd.DataFrame(columns=['minute', 'co2'])


# --- Langkah 2.2: Proses Data Cuaca ---
print("Memproses data cuaca...")
weather_folder_path = 'dataset/data-raw/cuaca'
# Membuat direktori jika belum ada
os.makedirs(weather_folder_path, exist_ok=True)
# Menambahkan file CSV dummy jika direktori kosong
if not glob.glob(os.path.join(weather_folder_path, "*.csv")):
    dummy_weather_data = {
        'timestamp': pd.to_datetime(['2025-04-24 00:00:00', '2025-04-24 00:01:00']),
        'temperature': [25.0, 25.1], 'humidity': [80, 81], 'rainfall': [0,0], 'pyrano': [0,0],
        'direction': [0,0], 'angle': [0,0], 'wind_speed': [0,0] # Kolom yang akan dihapus
    }
    dummy_weather_df = pd.DataFrame(dummy_weather_data)
    dummy_weather_df.to_csv(os.path.join(weather_folder_path, 'dummy_weather.csv'), index=False)
    print(f"File CSV dummy 'dummy_weather.csv' dibuat di {weather_folder_path}")

all_weather_files = sorted([f for f in os.listdir(weather_folder_path) if f.endswith('.csv')])
if all_weather_files:
    df_weather_list = []
    for f in all_weather_files:
        try:
            df_temp = pd.read_csv(os.path.join(weather_folder_path, f))
            if 'timestamp' in df_temp.columns:
                df_weather_list.append(df_temp)
            else:
                print(f"Peringatan: File {f} tidak memiliki kolom 'timestamp'. Dilewati.")
        except Exception as e:
            print(f"Error membaca file {f}: {e}. Dilewati.")

    if df_weather_list:
        df_weather = pd.concat(df_weather_list, ignore_index=True)
        df_weather = df_weather.drop(columns=['direction', 'angle', 'wind_speed'], errors='ignore')
        df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'], errors='coerce')
        df_weather.dropna(subset=['timestamp'], inplace=True) # Hapus baris jika timestamp NaN

        if not df_weather.empty:
            # Sample data cuaca per menit
            start_date_weather, end_date_weather = '2025-04-24', '2025-05-07'
            full_weather_range = pd.date_range(start=f'{start_date_weather} 00:00:00', end=f'{end_date_weather} 23:59:00', freq='T')
            full_weather_df = pd.DataFrame({'timestamp': full_weather_range})
            # Pastikan kolom yang ada di df_weather sebelum merge
            cols_to_merge = ['timestamp'] + [col for col in ['temperature', 'humidity', 'rainfall', 'pyrano'] if col in df_weather.columns]
            weather_sampled = pd.merge(full_weather_df, df_weather[cols_to_merge], on='timestamp', how='left')
            print("Data cuaca di-sample per menit.")
        else:
            print("Peringatan: DataFrame cuaca kosong setelah pembersihan awal.")
            weather_sampled = pd.DataFrame({'timestamp': pd.date_range(start='2025-04-24', end='2025-05-07 23:59:00', freq='T')}) # Hanya timestamp
    else:
        print("Peringatan: Tidak ada file cuaca yang valid yang dapat dibaca.")
        weather_sampled = pd.DataFrame({'timestamp': pd.date_range(start='2025-04-24', end='2025-05-07 23:59:00', freq='T')})
else:
    print(f"Peringatan: Tidak ada file CSV yang ditemukan di {weather_folder_path}.")
    weather_sampled = pd.DataFrame({'timestamp': pd.date_range(start='2025-04-24', end='2025-05-07 23:59:00', freq='T')})


# --- Langkah 2.3: Gabungkan Data CO2 dan Cuaca ---
print("Menggabungkan data CO2 dan cuaca...")
if not co2_sampled.empty and 'minute' in co2_sampled.columns:
    co2_sampled.rename(columns={'minute': 'timestamp'}, inplace=True)
    co2_sampled['timestamp'] = pd.to_datetime(co2_sampled['timestamp'])
else:
    print("Peringatan: co2_sampled kosong atau tidak memiliki kolom 'minute'. Membuat DataFrame co2_sampled kosong dengan kolom 'timestamp'.")
    co2_sampled = pd.DataFrame(columns=['timestamp', 'co2']) 

if 'timestamp' not in weather_sampled.columns:
    print("Peringatan: weather_sampled tidak memiliki kolom 'timestamp'. Membuatnya.")
    weather_sampled['timestamp'] = pd.date_range(start='2025-04-24', end='2025-05-07 23:59:00', freq='T')
weather_sampled['timestamp'] = pd.to_datetime(weather_sampled['timestamp'])

merged_df_temp = pd.merge(weather_sampled, co2_sampled, on='timestamp', how='left')

expected_columns_order = ['timestamp', 'co2', 'temperature', 'humidity', 'rainfall', 'pyrano']
for col in expected_columns_order:
    if col not in merged_df_temp.columns:
        print(f"Peringatan: Kolom '{col}' tidak ada di merged_df_temp setelah merge. Menambahkan sebagai NaN.")
        merged_df_temp[col] = np.nan 

merged_df = merged_df_temp[expected_columns_order]

print("Penggabungan selesai. Bentuk data gabungan:", merged_df.shape)
if merged_df.empty:
    print("Peringatan: DataFrame gabungan kosong. Periksa langkah pemrosesan data CO2 dan Cuaca.")
    default_cols = ['timestamp', 'co2', 'temperature', 'humidity', 'rainfall', 'pyrano']
    merged_df = pd.DataFrame(columns=default_cols)
    merged_df['timestamp'] = pd.date_range(start='2025-04-24', end='2025-05-07 23:59:00', freq='T')
    print(f"DataFrame gabungan default dibuat dengan bentuk: {merged_df.shape}")


df = merged_df.copy() 
print("--- Selesai: 2. Pemuatan Data ---")


# ==============================================================================
# 3. ANALISIS DATA EKSPLORATIF (EDA) & PRA-PEMROSESAN
# ==============================================================================
print("\n--- Memulai: 3. EDA & Pra-pemrosesan ---")

# --- Langkah 3.1: Inspeksi Data Awal ---
print("Struktur data awal (df dari langkah 2):") 
if not df.empty:
    df.info()
else:
    print("DataFrame df kosong. Tidak dapat menampilkan info.")


# --- Langkah 3.2: Analisis Nilai yang Hilang ---
print("\nNilai yang hilang per kolom (%):")
if not df.empty:
    print((df.isnull().sum() / len(df) * 100).round(2))
else:
    print("DataFrame df kosong. Tidak dapat menghitung nilai yang hilang.")


if not df.empty:
    # plt.figure(figsize=(12, 4))
    # sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    # plt.title('Heatmap Nilai yang Hilang Sebelum Penanganan')
    # plt.show()
    print("Visualisasi heatmap nilai yang hilang dikomentari.")
else:
    print("Tidak dapat membuat heatmap nilai yang hilang karena DataFrame kosong.")

# --- Langkah 3.3: Tangani Nilai yang Hilang ---
if not df.empty:
    df_filled = df.ffill().bfill() 
    print("\nNilai yang hilang setelah ffill/bfill:")
    print(df_filled.isnull().sum()) 
else:
    print("DataFrame df kosong. Penanganan nilai yang hilang dilewati.")
    df_filled = df.copy() 

# --- Langkah 3.4: Analisis Distribusi dan Outlier ---
print("Memvisualisasikan distribusi fitur dan outlier...")
num_cols = ['co2', 'temperature', 'humidity', 'rainfall', 'pyrano'] 
if not df_filled.empty:
    for col in num_cols:
        if col in df_filled.columns and not df_filled[col].isnull().all(): 
            # plt.figure(figsize=(14, 4))
            # plt.subplot(1, 2, 1)
            # sns.histplot(df_filled[col].dropna(), kde=True, bins=30) 
            # plt.title(f'Distribusi {col}')
            # plt.subplot(1, 2, 2)
            # sns.boxplot(x=df_filled[col].dropna()) 
            # plt.title(f'Boxplot {col}')
            # plt.tight_layout()
            # plt.show()
            print(f"Visualisasi distribusi dan boxplot untuk {col} dikomentari.")
        else:
            print(f"Peringatan: Kolom '{col}' tidak ada di df_filled atau semua nilainya NaN. Visualisasi dilewati.")
else:
    print("DataFrame df_filled kosong. Analisis distribusi dan outlier dilewati.")

# --- Langkah 3.5: Analisis Korelasi ---
print("Memvisualisasikan matriks korelasi fitur...")
if not df_filled.empty:
    numeric_cols_for_corr = [col for col in num_cols if col in df_filled.columns and pd.api.types.is_numeric_dtype(df_filled[col]) and not df_filled[col].isnull().all()]
    if numeric_cols_for_corr:
        # plt.figure(figsize=(8, 6))
        correlation_matrix = df_filled[numeric_cols_for_corr].corr() 
        # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        # plt.title('Matriks Korelasi Fitur')
        # plt.show()
        print("Visualisasi heatmap korelasi dikomentari.")
    else:
        print("Peringatan: Tidak ada kolom numerik yang valid untuk analisis korelasi.")
else:
    print("DataFrame df_filled kosong. Analisis korelasi dilewati.")

# --- Langkah 3.6: Visualisasi Time Series ---
print("Memvisualisasikan pola time series...")
if not df_filled.empty and 'timestamp' in df_filled.columns:
    if not pd.api.types.is_datetime64_any_dtype(df_filled['timestamp']):
        df_filled['timestamp'] = pd.to_datetime(df_filled['timestamp'])
        
    df_time_indexed = df_filled.set_index('timestamp')
    numeric_cols_for_plot = [col for col in num_cols if col in df_time_indexed.columns and pd.api.types.is_numeric_dtype(df_time_indexed[col]) and not df_time_indexed[col].isnull().all()]
    if numeric_cols_for_plot:
        # df_time_indexed[numeric_cols_for_plot].plot(subplots=True, figsize=(15, 10), title='Time Series Fitur')
        # plt.xlabel('Timestamp')
        # plt.tight_layout()
        # plt.show()
        print("Visualisasi time series fitur dikomentari.")
    else:
        print("Peringatan: Tidak ada kolom numerik yang valid untuk visualisasi time series.")
else:
    print("DataFrame df_filled kosong atau tidak memiliki kolom 'timestamp'. Visualisasi time series dilewati.")


# --- Langkah 3.7: Normalisasi Data ---
print("Normalisasi data menggunakan MinMaxScaler...")
# --- PERUBAHAN KUNCI DI SINI: Definisi cols_to_scale dipindahkan ke atas ---
cols_to_scale = ['co2', 'temperature', 'humidity', 'rainfall', 'pyrano']
df_scaled_intermediate = df_filled.copy() 
scalable_cols_in_df = [col for col in cols_to_scale if col in df_scaled_intermediate.columns and not df_scaled_intermediate[col].isnull().all()]

if not df_scaled_intermediate.empty and scalable_cols_in_df:
    if df_scaled_intermediate[scalable_cols_in_df].isnull().any().any():
        print("Peringatan: NaN ditemukan di kolom yang akan di-scale SETELAH ffill/bfill. Ini tidak diharapkan.")
    
    scaler_instance = MinMaxScaler()
    df_scaled_intermediate[scalable_cols_in_df] = scaler_instance.fit_transform(df_scaled_intermediate[scalable_cols_in_df])
    print("Kolom fitur telah dinormalisasi.")
    
    for col in cols_to_scale:
        if col not in scalable_cols_in_df: 
            if col not in df_scaled_intermediate.columns:
                print(f"Peringatan: Kolom '{col}' untuk scaling tidak ada di DataFrame. Membuat kolom dengan nilai default 0.0.")
                df_scaled_intermediate[col] = 0.0 
            elif df_scaled_intermediate[col].isnull().all(): 
                print(f"Peringatan: Kolom '{col}' semua nilainya NaN setelah ffill/bfill. Diisi dengan 0.0 sebelum reset_index.")
                df_scaled_intermediate[col] = 0.0

    df_scaled = df_scaled_intermediate.reset_index() 
    
    print("Data ternormalisasi dan df_scaled.reset_index() diterapkan (5 baris pertama):")
    print(df_scaled.head())

elif df_scaled_intermediate.empty:
    print("DataFrame df_filled kosong. Normalisasi dilewati.")
    df_scaled = df_filled.copy() 
    if not df_scaled.empty: 
        df_scaled = df_scaled.reset_index()
else: 
    print("Tidak ada kolom yang valid untuk dinormalisasi.")
    df_scaled = df_scaled_intermediate.copy() 
    for col_to_add in cols_to_scale: # Menggunakan variabel yang berbeda untuk menghindari konflik
        if col_to_add not in df_scaled.columns: 
            df_scaled[col_to_add] = 0.0
    if not df_scaled.empty: 
        df_scaled = df_scaled.reset_index()
# --- AKHIR PERUBAHAN ---

df_model_input = df_scaled.copy() 
print("--- Selesai: 3. EDA & Pra-pemrosesan ---")


# ==============================================================================
# 4. PEMODELAN MACHINE LEARNING
# ==============================================================================
print("\n--- Memulai: 4. Pemodelan Machine Learning ---")

# --- Langkah 4.1: Penyesuaian Timestamp dan Indeks ---
if not df_model_input.empty and 'timestamp' in df_model_input.columns:
    if not pd.api.types.is_datetime64_any_dtype(df_model_input['timestamp']):
        df_model_input['timestamp'] = pd.to_datetime(df_model_input['timestamp'])
    df_model_input = df_model_input.set_index('timestamp')
    print("Timestamp diatur sebagai indeks. Kolom 'index' tetap ada (sesuai notebook).")
elif df_model_input.empty:
    print("Peringatan: df_model_input kosong. Langkah pemodelan mungkin gagal.")
    min_cols = ['index','timestamp', 'co2', 'temperature', 'humidity', 'rainfall', 'pyrano'] 
    df_model_input = pd.DataFrame(columns=min_cols)
    df_model_input['timestamp'] = pd.date_range(start='2025-04-24', end='2025-05-07 23:59:00', freq='T')
    df_model_input['index'] = range(len(df_model_input)) 
    for col in ['co2', 'temperature', 'humidity', 'rainfall', 'pyrano']:
        df_model_input[col] = 0 
    df_model_input = df_model_input.set_index('timestamp')
    print("df_model_input default dibuat dengan kolom 'index'.")
else: 
    print("Peringatan: Kolom 'timestamp' tidak ditemukan di df_model_input.")
    if 'index' not in df_model_input.columns and len(df_model_input.index) == len(df_model_input): 
        df_model_input = df_model_input.reset_index() 

# --- Langkah 4.2: Rekayasa Fitur ---
print("Melakukan rekayasa fitur...")
target_column = 'co2'
exog_features = ['temperature', 'humidity', 'rainfall', 'pyrano'] 

if target_column not in df_model_input.columns:
    print(f"Peringatan: Kolom target '{target_column}' tidak ada. Membuatnya dengan nilai default 0.")
    df_model_input[target_column] = 0.0 
for col in exog_features:
    if col not in df_model_input.columns:
        print(f"Peringatan: Fitur eksogen '{col}' tidak ada. Membuatnya dengan nilai default 0.")
        df_model_input[col] = 0.0

df_model_input['co2_target_1min'] = df_model_input[target_column].shift(-1)
df_model_input['co2_target_1hour'] = df_model_input[target_column].shift(-60)

n_lags = 10
lag_cols = []
for i in range(1, n_lags + 1):
    col_name = f'{target_column}_lag_{i}'
    df_model_input[col_name] = df_model_input[target_column].shift(i)
    lag_cols.append(col_name)

feature_columns = lag_cols + exog_features 
print(f"Fitur yang digunakan untuk pemodelan: {feature_columns}")

cols_for_dropna_1min = ['co2_target_1min'] + feature_columns
cols_for_dropna_1hour = ['co2_target_1hour'] + feature_columns

missing_cols_1min = [col for col in cols_for_dropna_1min if col not in df_model_input.columns]
if missing_cols_1min:
    print(f"Peringatan: Kolom berikut hilang untuk dropna (1 menit): {missing_cols_1min}. Membuatnya dengan NaN.")
    for col in missing_cols_1min: df_model_input[col] = np.nan

missing_cols_1hour = [col for col in cols_for_dropna_1hour if col not in df_model_input.columns]
if missing_cols_1hour:
    print(f"Peringatan: Kolom berikut hilang untuk dropna (1 jam): {missing_cols_1hour}. Membuatnya dengan NaN.")
    for col in missing_cols_1hour: df_model_input[col] = np.nan


df_1min = df_model_input.dropna(subset=cols_for_dropna_1min).copy()
actual_feature_columns_1min = [col for col in feature_columns if col in df_1min.columns]
X_1min_all = df_1min[actual_feature_columns_1min] if not df_1min.empty and actual_feature_columns_1min else pd.DataFrame(columns=feature_columns)
y_1min_all = df_1min['co2_target_1min'] if not df_1min.empty and 'co2_target_1min' in df_1min else pd.Series(dtype='float64')

df_1hour = df_model_input.dropna(subset=cols_for_dropna_1hour).copy()
actual_feature_columns_1hour = [col for col in feature_columns if col in df_1hour.columns]
X_1hour_all = df_1hour[actual_feature_columns_1hour] if not df_1hour.empty and actual_feature_columns_1hour else pd.DataFrame(columns=feature_columns)
y_1hour_all = df_1hour['co2_target_1hour'] if not df_1hour.empty and 'co2_target_1hour' in df_1hour else pd.Series(dtype='float64')


print(f"Data siap untuk model 1 menit: {X_1min_all.shape}")
print(f"Data siap untuk model 1 jam: {X_1hour_all.shape}")

if X_1min_all.empty or y_1min_all.empty:
    print("Peringatan: Tidak ada data yang tersisa untuk model 1 menit setelah rekayasa fitur dan dropna.")
if X_1hour_all.empty or y_1hour_all.empty:
    print("Peringatan: Tidak ada data yang tersisa untuk model 1 jam setelah rekayasa fitur dan dropna.")


# --- Langkah 4.3: Pembagian Data (Latih/Uji) ---
print("Membagi data menjadi set latih dan uji...")
if not X_1min_all.empty and not y_1min_all.empty:
    X_train_1min, X_test_1min, y_train_1min, y_test_1min = train_test_split(
        X_1min_all, y_1min_all, test_size=0.2, shuffle=False)
else:
    X_train_1min, X_test_1min, y_train_1min, y_test_1min = pd.DataFrame(), pd.DataFrame(), pd.Series(dtype='float64'), pd.Series(dtype='float64')

if not X_1hour_all.empty and not y_1hour_all.empty:
    X_train_1hour, X_test_1hour, y_train_1hour, y_test_1hour = train_test_split(
        X_1hour_all, y_1hour_all, test_size=0.2, shuffle=False)
else:
    X_train_1hour, X_test_1hour, y_train_1hour, y_test_1hour = pd.DataFrame(), pd.DataFrame(), pd.Series(dtype='float64'), pd.Series(dtype='float64')

print(f"Pembagian Latih/Uji (1 menit): {len(X_train_1min)}/{len(X_test_1min)}")
print(f"Pembagian Latih/Uji (1 jam): {len(X_train_1hour)}/{len(X_test_1hour)}")


# --- Langkah 4.4: Pelatihan Model, Evaluasi, dan Perbandingan ---
evaluation_results_storage = {"1_min": {}, "1_hour": {}}

def plot_actual_vs_predicted(y_true, y_pred, model_name, horizon_tag, test_index):
    print(f"Plot aktual vs prediksi untuk {model_name} ({horizon_tag}) dikomentari.")
    pass


def train_evaluate_models_set(X_train, y_train, X_test, y_test, horizon_tag):
    models_definition = {
        "SVM Regressor": SVR(kernel='rbf'),
        "RandomForest": RandomForestRegressor(random_state=42, n_estimators=100),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=100)
    }
    
    current_results = {}
    
    if X_train.empty or y_train.empty or X_test.empty or y_test.empty:
        print(f"Data latih atau uji kosong untuk horizon {horizon_tag}. Pelatihan model dilewati.")
        for name in models_definition.keys():
            current_results[name] = {"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "R2": np.nan}
        return current_results

    for name, model_instance in models_definition.items():
        print(f"\nMelatih {name} untuk prediksi {horizon_tag}...")
        try:
            model_instance.fit(X_train, y_train)
            
            print(f"Mengevaluasi {name}...")
            predictions = model_instance.predict(X_test)
            
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            
            current_results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
            
            print(f"Hasil untuk {name} ({horizon_tag}): MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
            plot_actual_vs_predicted(y_test, predictions, name, horizon_tag, X_test.index) 
        except Exception as e:
            print(f"Error saat melatih atau mengevaluasi {name} untuk {horizon_tag}: {e}")
            current_results[name] = {"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "R2": np.nan}
            
    return current_results

print("\n--- Pelatihan dan Evaluasi untuk Horizon 1 Menit ---")
if not X_train_1min.empty:
    evaluation_results_storage["1_min"] = train_evaluate_models_set(X_train_1min, y_train_1min, X_test_1min, y_test_1min, "1 Menit")
else:
    print("Data latih untuk horizon 1 menit kosong. Pelatihan dilewati.")
    evaluation_results_storage["1_min"] = {model_name: {"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "R2": np.nan} for model_name in ["SVM Regressor", "RandomForest", "XGBoost"]}

print("\n--- Pelatihan dan Evaluasi untuk Horizon 1 Jam ---")
if not X_train_1hour.empty:
    evaluation_results_storage["1_hour"] = train_evaluate_models_set(X_train_1hour, y_train_1hour, X_test_1hour, y_test_1hour, "1 Jam")
else:
    print("Data latih untuk horizon 1 jam kosong. Pelatihan dilewati.")
    evaluation_results_storage["1_hour"] = {model_name: {"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "R2": np.nan} for model_name in ["SVM Regressor", "RandomForest", "XGBoost"]}


# --- Langkah 4.5: Bandingkan Model dan Pilih yang Terbaik ---
def display_detailed_comparison(eval_results_dict, horizon_tag_display):
    if not eval_results_dict:
        print(f"Tidak ada hasil evaluasi untuk {horizon_tag_display}.")
        return None
    
    eval_df = pd.DataFrame(eval_results_dict).T
    
    if eval_df.empty:
        print(f"Tidak ada model yang dievaluasi untuk {horizon_tag_display}.")
        return None

    print(f"\n--- Perbandingan Model Keseluruhan untuk Horizon {horizon_tag_display} ---")
    print(eval_df.to_string())
    
    eval_df_valid = eval_df.dropna(how='all', subset=["MAE", "MSE", "RMSE", "R2"])

    if eval_df_valid.empty:
        print(f"Tidak ada model yang berhasil dievaluasi (semua metrik NaN) untuk {horizon_tag_display}.")
        return None

    metrics_criteria_ranking = {
        "R2": {"rank_ascending": False}, 
        "MAE": {"rank_ascending": True},  
        "MSE": {"rank_ascending": True},
        "RMSE": {"rank_ascending": True}
    }
    
    model_composite_scores = pd.Series(0.0, index=eval_df_valid.index) 
    num_valid_models = len(eval_df_valid)
    
    for metric_name, config_ranking in metrics_criteria_ranking.items():
        if metric_name in eval_df_valid.columns:
            ranks = eval_df_valid[metric_name].rank(method='min', ascending=config_ranking["rank_ascending"], na_option='bottom')
            scores_for_metric = num_valid_models - ranks + 1
            model_composite_scores = model_composite_scores.add(scores_for_metric.fillna(0), fill_value=0)
        
    print("\nSkor Komposit Model (lebih tinggi lebih baik):")
    sorted_scores = model_composite_scores.sort_values(ascending=False)
    print(sorted_scores.to_string())
    
    if not sorted_scores.empty:
        best_overall_model_name_selected = sorted_scores.idxmax()
        if sorted_scores.max() > 0:
             print(f"-> Model Terbaik Keseluruhan untuk {horizon_tag_display}: {best_overall_model_name_selected}")
             return best_overall_model_name_selected
        else:
            print(f"Tidak dapat menentukan model terbaik untuk {horizon_tag_display} karena semua skor nol atau negatif.")
            return None
    else:
        print(f"Tidak dapat menentukan model terbaik untuk {horizon_tag_display} karena tidak ada skor komposit.")
        return None

best_model_1min_name_final = display_detailed_comparison(evaluation_results_storage["1_min"], "1 Menit")
best_model_1hour_name_final = display_detailed_comparison(evaluation_results_storage["1_hour"], "1 Jam")


# --- Langkah 4.6: Latih Ulang dan Simpan Model Terbaik ---
print("\n--- Pelatihan Ulang dan Penyimpanan Model Final ---")

model_save_dir_path = "saved_models"
if not os.path.exists(model_save_dir_path):
    os.makedirs(model_save_dir_path)
    print(f"Direktori dibuat: {model_save_dir_path}")

def retrain_and_save_final_model(model_name_to_save, X_all_data, y_all_data, horizon_tag_for_file):
    if model_name_to_save is None:
        print(f"Penyimpanan dilewati untuk horizon {horizon_tag_for_file} karena tidak ada model terbaik yang ditentukan.")
        return

    if X_all_data.empty or y_all_data.empty:
        print(f"Data untuk pelatihan ulang model {model_name_to_save} ({horizon_tag_for_file}) kosong. Penyimpanan dilewati.")
        return

    print(f"Melatih ulang model final '{model_name_to_save}' untuk horizon {horizon_tag_for_file} pada semua data yang tersedia...")
    
    final_model_instance = None
    if model_name_to_save == "SVM Regressor":
        final_model_instance = SVR(kernel='rbf')
    elif model_name_to_save == "RandomForest":
        final_model_instance = RandomForestRegressor(random_state=42, n_estimators=100)
    elif model_name_to_save == "XGBoost":
        final_model_instance = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=100)
    else:
        print(f"Error: Nama model tidak dikenal '{model_name_to_save}'.")
        return

    try:
        final_model_instance.fit(X_all_data, y_all_data)
        
        model_filename_to_save = f"final_model_{model_name_to_save.replace(' ', '_').lower()}_{horizon_tag_for_file}.joblib"
        model_full_path = os.path.join(model_save_dir_path, model_filename_to_save)
        
        joblib.dump(final_model_instance, model_full_path)
        print(f"Model berhasil disimpan ke: {model_full_path}")
    except Exception as e:
        print(f"Error saat melatih ulang atau menyimpan model {model_name_to_save} untuk {horizon_tag_for_file}: {e}")

retrain_and_save_final_model(best_model_1min_name_final, X_1min_all, y_1min_all, "1_menit")
retrain_and_save_final_model(best_model_1hour_name_final, X_1hour_all, y_1hour_all, "1_jam")

print("\n--- Akhir Proses ---")