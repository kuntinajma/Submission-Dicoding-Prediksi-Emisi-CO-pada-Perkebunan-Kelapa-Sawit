# -*- coding: utf-8 -*-
"""
main.py

Script utama yang mengintegrasikan seluruh alur kerja machine learning,
mulai dari pemuatan dan penggabungan data, analisis data eksplorasi (EDA)
dan pra-pemrosesan, hingga pelatihan, evaluasi, dan penyimpanan model.

Alur Kerja:
1.  **Pemuatan & Penggabungan Data**: Menjalankan logika dari `data_load_merge.py`.
    - Input: File CSV mentah dari `dataset/data-raw/`.
    - Output: DataFrame gabungan di memori.
2.  **EDA & Pra-pemrosesan**: Menjalankan logika dari `eda_prepo.py`.
    - Input: DataFrame gabungan dari langkah sebelumnya.
    - Output: DataFrame yang sudah bersih dan ternormalisasi di memori.
3.  **Pemodelan Machine Learning**: Menjalankan logika dari `model_ml.py`.
    - Input: DataFrame yang sudah dipra-pemrosesan.
    - Output: Model machine learning yang telah dilatih dan disimpan dalam format .joblib
      serta hasil evaluasi yang dicetak ke konsol.
"""

# =============================================================================
# BAGIAN 0: IMPORT SEMUA LIBRARY YANG DIBUTUHKAN
# =============================================================================
import os
import pandas as pd
import numpy as np
import glob
from statistics import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Nonaktifkan peringatan yang tidak relevan jika diperlukan
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# BAGIAN 1: PEMUATAN DAN PENGGABUNGAN DATA (Logika dari data_load_merge.py)
# =============================================================================
def run_data_loading_and_merge():
    """
    Menggabungkan data mentah CO2 dan cuaca, melakukan resampling per menit,
    dan menghasilkan satu DataFrame tunggal yang siap untuk pra-pemrosesan.
    """
    print("="*50)
    print("MEMULAI TAHAP 1: Pemuatan dan Penggabungan Data")
    print("="*50)

    # --- Membaca dan Menggabungkan Data CO2 ---
    print("\n[1.1] Memproses data CO2...")
    folder_path_co2 = 'dataset/data-raw/co2'
    all_csv_files = glob.glob(os.path.join(folder_path_co2, "*.csv"))
    df_co2 = pd.concat([pd.read_csv(f) for f in all_csv_files], ignore_index=True)
    df_co2 = df_co2[['timestamp', 'co2']]
    df_co2['timestamp'] = pd.to_datetime(df_co2['timestamp'], errors='coerce')
    df_co2['co2'] = pd.to_numeric(df_co2['co2'], errors='coerce')

    # --- Sampling dan Data CO2 ---
    print("[1.2] Melakukan resampling data CO2 per menit...")
    df_co2['minute'] = df_co2['timestamp'].dt.floor('min')
    minute_co2 = df_co2.groupby('minute')['co2'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    ).reset_index()
    full_minutes = pd.date_range(
        start=df_co2['minute'].min().floor('D'),
        end=df_co2['minute'].max().ceil('D') - pd.Timedelta(minutes=1),
        freq='1min'
    )
    full_minutes_df = pd.DataFrame({'minute': full_minutes})
    co2_sampled = full_minutes_df.merge(minute_co2, on='minute', how='left')

    # --- Memuat dan Membersihkan Data Cuaca ---
    print("[1.3] Memproses data cuaca...")
    folder_path_weather = 'dataset/data-raw/cuaca'
    file_list_weather = sorted(os.listdir(folder_path_weather))
    df_list_weather = []
    for file in file_list_weather:
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder_path_weather, file))
            df_list_weather.append(df)
    df_weather = pd.concat(df_list_weather, ignore_index=True)
    df_weather = df_weather.drop(['direction', 'angle', 'wind_speed'], axis=1)
    df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])

    # --- Memfilter dan Merapikan Data Cuaca ---
    print("[1.4] Melakukan filtering dan alignment data cuaca...")
    start_date = '2025-04-24'
    end_date = '2025-05-07'
    filtered_weather = df_weather[(df_weather['timestamp'] >= start_date) & (df_weather['timestamp'] <= end_date)].copy()
    full_range = pd.date_range(start=start_date + ' 00:00:00', end=end_date + ' 23:59:00', freq='min')
    full_weather_df = pd.DataFrame({'timestamp': full_range})
    weather_sampled = pd.merge(full_weather_df, filtered_weather, on='timestamp', how='left')

    # --- Menggabungkan Data CO2 dan Cuaca ---
    print("[1.5] Menggabungkan data CO2 dan cuaca...")
    cuaca_df = weather_sampled
    co2_df = co2_sampled
    co2_df['minute'] = pd.to_datetime(co2_df['minute'])
    cuaca_df['timestamp'] = pd.to_datetime(cuaca_df['timestamp'])
    merged_df = pd.merge(co2_df, cuaca_df, left_on='minute', right_on='timestamp', how='left')
    merged_df.drop(columns=['timestamp'], inplace=True)
    merged_df.rename(columns={'minute': 'timestamp'}, inplace=True)
    
    # Menyimpan output final dari tahap ini untuk checkpointing
    output_merged_path = 'dataset/data-clean/data_output_collecting.csv'
    os.makedirs(os.path.dirname(output_merged_path), exist_ok=True)
    merged_df.to_csv(output_merged_path, index=False)
    print(f"-> Data gabungan berhasil disimpan di: {output_merged_path}")
    print("\nTAHAP 1 SELESAI.")
    
    return merged_df


# =============================================================================
# BAGIAN 2: EDA DAN PRA-PEMROSESAN (Logika dari eda_prepo.py)
# =============================================================================
def run_eda_and_preprocessing(input_df):
    """
    Melakukan analisis data eksplorasi, menangani missing values,
    dan melakukan normalisasi data.
    """
    print("\n" + "="*50)
    print("MEMULAI TAHAP 2: EDA dan Pra-pemrosesan")
    print("="*50)
    
    df = input_df.copy()

    # --- Konversi Tipe Data dan Inspeksi Awal ---
    print("\n[2.1] Inspeksi data awal...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Jumlah baris awal: {len(df)}")
    print("Jumlah missing values awal per kolom:")
    print(df.isnull().sum())

    # --- Visualisasi Pola Missing Values (Opsional, di-comment untuk eksekusi script) ---
    # print("\n[2.2] Membuat heatmap missing values...")
    # plt.figure(figsize=(15, 5))
    # sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    # plt.title('Heatmap Missing Values per Kolom')
    # plt.show()

    # --- Penanganan Missing Values ---
    print("\n[2.2] Menangani missing values dengan ffill dan bfill...")
    df_filled = df.ffill().bfill()
    print("Jumlah missing values setelah penanganan:")
    print(df_filled.isnull().sum())

    # --- Deteksi dan Investigasi Outlier ---
    print("\n[2.3] Investigasi outlier pada kolom CO2...")
    # Mengisolasi outlier potensial (berdasarkan analisis di notebook)
    outlier_time = pd.to_datetime('2025-05-02 01:15:00')
    df_filled_indexed = df_filled.set_index('timestamp')
    outlier_value = df_filled_indexed.loc[outlier_time, 'co2'] if outlier_time in df_filled_indexed.index else "Tidak ditemukan"
    print(f"Nilai pada waktu outlier (2025-05-02 01:15:00): {outlier_value}")
    print("Berdasarkan analisis notebook, outlier ini tidak ekstrem dan diputuskan untuk dipertahankan.")

    # --- Normalisasi Data ---
    print("\n[2.4] Melakukan normalisasi data (Min-Max Scaling)...")
    scaler = MinMaxScaler()
    cols_to_scale = ['co2', 'temperature', 'humidity', 'rainfall', 'pyrano']
    df_scaled = df_filled.copy()
    
    # Pastikan kolom timestamp tidak jadi index sebelum scaling
    if 'timestamp' in df_scaled.columns:
        df_scaled_data = df_scaled[cols_to_scale]
    else:
        df_scaled_data = df_scaled.reset_index()[cols_to_scale]

    df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled_data)
    
    # Jika timestamp adalah index, reset index agar kembali jadi kolom
    if isinstance(df_scaled.index, pd.DatetimeIndex):
       df_scaled = df_scaled.reset_index()

    # Menyimpan output final dari tahap ini untuk checkpointing
    output_scaled_path = 'dataset/data-clean/data_scaled.csv'
    os.makedirs(os.path.dirname(output_scaled_path), exist_ok=True)
    df_scaled.to_csv(output_scaled_path, index=False)
    print(f"-> Data yang sudah dinormalisasi disimpan di: {output_scaled_path}")
    print("\nTAHAP 2 SELESAI.")

    return df_scaled


# =============================================================================
# BAGIAN 3: PEMODELAN MACHINE LEARNING (Logika dari model_ml.py)
# =============================================================================
def run_modeling(input_df):
    """
    Melakukan feature engineering, membagi data, melatih beberapa model regresi,
    mengevaluasi, membandingkan, dan menyimpan model terbaik.
    """
    print("\n" + "="*50)
    print("MEMULAI TAHAP 3: Pemodelan Machine Learning")
    print("="*50)
    
    df = input_df.copy()

    # --- Penyesuaian Indeks Waktu ---
    print("\n[3.1] Menyiapkan data dan indeks waktu...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # --- Rekayasa Fitur (Feature Engineering) ---
    print("[3.2] Melakukan rekayasa fitur (lag features)...")
    target_column = 'co2'
    exog_features = ['temperature', 'humidity', 'rainfall', 'pyrano']
    df['co2_target_1min'] = df[target_column].shift(-1)
    df['co2_target_1hour'] = df[target_column].shift(-60)
    n_lags = 10
    lag_cols = []
    for i in range(1, n_lags + 1):
        col_name = f'{target_column}_lag_{i}'
        df[col_name] = df[target_column].shift(i)
        lag_cols.append(col_name)
    feature_columns = lag_cols + exog_features

    # --- Membersihkan dan Memisahkan Data Final ---
    df_1min = df.dropna(subset=['co2_target_1min'] + feature_columns).copy()
    X_1min_all = df_1min[feature_columns]
    y_1min_all = df_1min['co2_target_1min']
    df_1hour = df.dropna(subset=['co2_target_1hour'] + feature_columns).copy()
    X_1hour_all = df_1hour[feature_columns]
    y_1hour_all = df_1hour['co2_target_1hour']

    # --- Pembagian Data (Train-Test Split) ---
    print("[3.3] Membagi data menjadi data latih dan uji...")
    X_train_1min, X_test_1min, y_train_1min, y_test_1min = train_test_split(
        X_1min_all, y_1min_all, test_size=0.2, shuffle=False
    )
    X_train_1hour, X_test_1hour, y_train_1hour, y_test_1hour = train_test_split(
        X_1hour_all, y_1hour_all, test_size=0.2, shuffle=False
    )

    # --- Persiapan Pelatihan dan Evaluasi ---
    evaluation_results_storage = {"1_min": {}, "1_hour": {}}
    trained_models_storage = {"1_min": {}, "1_hour": {}}

    # Helper function untuk plotting
    def plot_actual_vs_predicted(y_true, y_pred, model_name, horizon_tag, test_index):
        plt.figure(figsize=(15, 6))
        plt.plot(test_index, y_true, label='Nilai Aktual CO2', color='blue')
        plt.plot(test_index, y_pred, label=f'Prediksi {model_name}', color='red', linestyle='--')
        plt.title(f'Perbandingan Aktual vs Prediksi ({model_name}) - {horizon_tag}')
        plt.xlabel('Timestamp')
        plt.ylabel('Kadar CO2 (Scaled)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # Non-interactive mode, save the plot instead of showing
        plot_dir = "plots"
        os.makedirs(plot_dir, exist_ok=True)
        filename = f"{plot_dir}/plot_{model_name.replace(' ', '_')}_{horizon_tag.replace(' ', '_')}.png"
        plt.savefig(filename)
        print(f"-> Visualisasi disimpan di: {filename}")
        plt.close() # Close the plot to free memory

    # --- Loop Pelatihan dan Evaluasi Model ---
    models_to_train = {
        "SVM Regressor": SVR(kernel='rbf'),
        "RandomForest": RandomForestRegressor(random_state=42, n_estimators=100),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=100)
    }

    datasets = {
        "1 Menit": (X_train_1min, X_test_1min, y_train_1min, y_test_1min),
        "1 Jam": (X_train_1hour, X_test_1hour, y_train_1hour, y_test_1hour)
    }

    storage_keys = {"1 Menit": "1_min", "1 Jam": "1_hour"}

    for horizon_tag, (X_train, X_test, y_train, y_test) in datasets.items():
        print(f"\n--- Memproses Prediksi Horizon: {horizon_tag} ---")
        storage_key = storage_keys[horizon_tag]
        
        for model_name, model_instance in models_to_train.items():
            print(f"\n[3.4] Melatih model: {model_name} untuk {horizon_tag}")
            
            # Melatih model
            model_instance.fit(X_train, y_train)
            trained_models_storage[storage_key][model_name] = model_instance
            
            # Prediksi
            predictions = model_instance.predict(X_test)
            
            # Evaluasi
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            
            # Simpan hasil
            evaluation_results_storage[storage_key][model_name] = {
                "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2
            }
            
            print(f"Hasil Evaluasi {model_name} ({horizon_tag}):")
            print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")
            
            # Visualisasi
            plot_actual_vs_predicted(y_test, predictions, model_name, horizon_tag, X_test.index)

    # --- Perbandingan dan Pemilihan Model Final ---
    print("\n[3.5] Membandingkan model dan memilih yang terbaik...")

    # Helper function untuk perbandingan
    def display_detailed_comparison(eval_results_dict, horizon_tag):
        eval_df = pd.DataFrame(eval_results_dict).T
        print(f"\n{'-'*20} Tabel Performa Model ({horizon_tag}) {'-'*20}")
        print(eval_df[['MAE', 'MSE', 'RMSE', 'R2']].to_string())
        
        metrics_criteria = {
            "MAE": {"rank_ascending": True}, "MSE": {"rank_ascending": True},
            "RMSE": {"rank_ascending": True}, "R2": {"rank_ascending": False}
        }
        model_scores = pd.Series(0, index=eval_df.index)
        num_models = len(eval_df)
        for metric, criteria in metrics_criteria.items():
            ranks = eval_df[metric].rank(method='min', ascending=criteria["rank_ascending"])
            scores = num_models - ranks + 1
            model_scores = model_scores.add(scores, fill_value=0)
        
        best_overall_model_name = model_scores.idxmax()
        print(f"\nSkor Komposit Model ({horizon_tag}):\n{model_scores.sort_values(ascending=False).to_string()}")
        print(f"\n-> Model Terbaik untuk {horizon_tag}: {best_overall_model_name}")
        return best_overall_model_name

    best_model_1min_name = display_detailed_comparison(evaluation_results_storage["1_min"], "1 Menit")
    best_model_1hour_name = display_detailed_comparison(evaluation_results_storage["1_hour"], "1 Jam")
    
    # --- Pelatihan Ulang dan Penyimpanan Model Final ---
    print("\n[3.6] Melatih ulang dan menyimpan model final terpilih...")
    
    model_save_dir = "saved_models"
    os.makedirs(model_save_dir, exist_ok=True)

    # Helper function untuk melatih ulang
    def retrain_and_save_final_model(best_model_name, X_all, y_all, horizon_tag):
        print(f"\nMelatih ulang {best_model_name} untuk {horizon_tag} menggunakan seluruh data...")
        
        if best_model_name == "SVM Regressor": final_model = SVR(kernel='rbf')
        elif best_model_name == "RandomForest": final_model = RandomForestRegressor(random_state=42, n_estimators=100)
        elif best_model_name == "XGBoost": final_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=100)
        
        final_model.fit(X_all, y_all)
        
        model_filename = f"final_model_{best_model_name.replace(' ', '_').lower()}_{horizon_tag.replace(' ', '_').lower()}.joblib"
        model_path = os.path.join(model_save_dir, model_filename)
        joblib.dump(final_model, model_path)
        print(f"-> Model final disimpan di: {model_path}")

    # Eksekusi pelatihan ulang
    retrain_and_save_final_model(best_model_1min_name, X_1min_all, y_1min_all, "1 Menit")
    retrain_and_save_final_model(best_model_1hour_name, X_1hour_all, y_1hour_all, "1 Jam")
    
    print("\nTAHAP 3 SELESAI.")


# =============================================================================
# BLOK EKSEKUSI UTAMA (MAIN)
# =============================================================================
if __name__ == "__main__":
    print("==========================================================")
    print("            MEMULAI ALUR KERJA FORECASTING CO2            ")
    print("==========================================================")
    
    # Jalankan Tahap 1: Pemuatan Data
    # Fungsi ini akan mengembalikan DataFrame gabungan
    merged_data = run_data_loading_and_merge()
    
    # Jalankan Tahap 2: EDA dan Pra-pemrosesan
    # Fungsi ini mengambil DataFrame dari tahap 1 dan mengembalikan DataFrame yang sudah dinormalisasi
    scaled_data = run_eda_and_preprocessing(merged_data)
    
    # Jalankan Tahap 3: Pemodelan
    # Fungsi ini mengambil DataFrame dari tahap 2 untuk melatih dan menyimpan model
    run_modeling(scaled_data)
    
    print("\n==========================================================")
    print("        SELURUH PROSES TELAH BERHASIL DISELESAIKAN        ")
    print("==========================================================")