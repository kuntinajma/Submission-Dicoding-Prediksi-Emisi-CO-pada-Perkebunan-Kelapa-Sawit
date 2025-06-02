# <a name="_dzv9r6lawrr2"></a>**Laporan Proyek Machine Learning - Kunti Najma Jalia**
## <a name="_vrhp3mj70t5t"></a>**Domain Proyek - Prediksi Emisi CO₂ pada Perkebunan Kelapa Sawit**
Perkebunan kelapa sawit (*Elaeis guineensis Jacq*.) memainkan peran penting dalam perekonomian Indonesia, menjadikan negara ini sebagai salah satu produsen dan pengekspor minyak sawit mentah (CPO) terbesar di dunia (Arjuna & Santosa, 2018). Selain menyumbang devisa, sektor ini juga menjadi sumber penghidupan bagi jutaan masyarakat. Namun demikian, industri kelapa sawit kerap dikritik karena dampak lingkungannya, khususnya terkait emisi gas rumah kaca (GRK) seperti karbon dioksida (CO₂), yang berkontribusi besar terhadap pemanasan global (Hong, 2023). Emisi ini kerap digunakan sebagai alasan dalam kampanye negatif terhadap sawit di pasar global (Maisarah et al., 2024).

Sumber utama emisi CO₂ dalam perkebunan sawit berasal dari konversi lahan, terutama saat hutan atau lahan gambut diubah menjadi areal perkebunan (Putri et al., 2016). Proses ini menyebabkan hilangnya cadangan karbon dalam biomassa dan tanah, serta oksidasi gambut yang melepaskan CO₂ dalam jumlah besar (Arjuna & Santosa, 2018). Selain itu, penggunaan pupuk nitrogen berkontribusi terhadap pelepasan N₂O, gas rumah kaca yang jauh lebih kuat dibanding CO₂ (K. et al., 2024). Aktivitas lainnya seperti respirasi tanah, penggunaan bahan bakar untuk alat berat dan transportasi, serta pengolahan limbah cair pabrik sawit (POME), juga menyumbang pada peningkatan emisi karbon (Arjuna & Santosa, 2018).

Sebagai solusi, pendekatan berbasis teknologi menjadi sangat relevan. *Precision Agriculture Technologies* (PAT) berpotensi mengoptimalkan input pertanian untuk menekan emisi (K. et al., 2024), sementara *machine learning* (ML) dapat digunakan untuk membangun model prediktif berbasis data historis kondisi lingkungan dan cuaca (Bussaban et al., 2024). Dengan model prediksi yang akurat, pengelola dapat merancang strategi berbasis data untuk mengidentifikasi sumber emisi, meningkatkan efisiensi sumber daya, serta menyesuaikan praktik budidaya agar lebih berkelanjutan. Hal ini juga mendukung pelaporan emisi yang lebih transparan sesuai dengan standar nasional dan global (Arjuna & Santosa, 2018), serta meningkatkan daya saing kelapa sawit Indonesia di pasar internasional yang semakin peduli terhadap isu iklim.

**Referensi (Contoh):**

- Arjuna, R., & Santosa, E. (2018). Asesmen carbon footprint pada produksi minyak kelapa sawit (*Elaeis guineensis* Jacq.) di Kebun Sei Lukut, Kabupaten Siak, Riau. *Buletin Agrohorti*. 6(2): 287–295.
- Bussaban, K., Kularbphettong, K., Raksuntorn, N., & Boonseng, C. (2024). Prediction of CO₂ emissions using machine learning. *Edelweiss Applied Science and Technology*. 8(4): 1–11.
- Hong, W. O. (2023). Review on carbon footprint of the palm oil industry: Insights into recent developments. *International Journal of Sustainable Development and Planning*. 18(2): 447–455.
- K., F., Betül, R. S., & B., A. V. (2024). The importance and contribution of precision agriculture technologies in reducing greenhouse gas emissions. *International Journal of Advanced Research*. 12(12): 944–959.
- Maisarah, Dian, R., & Febrianto, E. B. (2024). Uji serapan karbon (sekuestrasi) tanaman kelapa sawit sebagai baseline pada trading karbon. *Ranah Research: Journal of Multidisciplinary Research and Development*. 7(1): 633–643.
- Putri, T. T. A., Syaufina, L., & Anshari, G. Z. (2016). Emisi karbon dioksida (CO₂) rizosfer dan non rizosfer dari perkebunan kelapa sawit (*Elaeis guineensis*) pada lahan gambut dangkal. *Jurnal Tanah dan Iklim*. 40(1): 43–50.
## <a name="_79f4lsa8teb3"></a>**Business Understanding**
Proses klarifikasi masalah dalam proyek ini melibatkan pemahaman mendalam terhadap kebutuhan untuk memprediksi emisi CO2 di perkebunan kelapa sawit dan bagaimana machine learning dapat memberikan solusi yang bernilai.
### <a name="_byy4wwk8nuud"></a>**Problem Statements**
1. Bagaimana cara memprediksi secara akurat jumlah emisi gas CO2 yang dihasilkan oleh perkebunan kelapa sawit berdasarkan data historis kondisi lingkungan dan cuaca untuk interval waktu 1 menit dan 1 jam ke depan?
1. Faktor-faktor cuaca dan lingkungan apa saja yang paling signifikan mempengaruhi tingkat emisi CO2 di area perkebunan kelapa sawit?
1. Algoritma machine learning manakah (antara Support Vector Machine, Random Forest, dan XGBoost) yang memberikan performa terbaik dalam memprediksi emisi CO2 untuk interval prediksi 1 menit dan 1 jam?
### <a name="_icljren7wknf"></a>**Goals**
1. Mengembangkan model regresi machine learning yang mampu memprediksi tingkat emisi CO2 dengan tingkat akurasi yang dapat diterima (berdasarkan R-squared atau RMSE) untuk prediksi 1 menit dan 1 jam.
1. Mengidentifikasi dan menganalisis fitur-fitur (variabel cuaca dan lingkungan) yang memiliki kontribusi terbesar terhadap prediksi emisi CO2.
1. Mengevaluasi dan membandingkan kinerja model Support Vector Regression (SVR), Random Forest Regressor, dan XGBoost Regressor untuk menentukan model prediksi CO2 yang paling optimal untuk setiap interval prediksi (1 menit dan 1 jam).
### <a name="_lrpvxhvr3ust"></a>**Solution statements**
1. Membangun dan melatih tiga model regresi machine learning (Support Vector Regression, Random Forest Regressor, XGBoost Regressor) untuk dua skenario prediksi: interval 1 menit dan interval 1 jam, menggunakan dataset historis emisi CO2 dan data cuaca yang telah diproses.
1. Melakukan optimasi hyperparameter untuk setiap model pada kedua skenario prediksi (misalnya menggunakan GridSearchCV atau RandomizedSearchCV) untuk meningkatkan performa prediksinya.
1. Mengevaluasi ketiga model tersebut pada kedua skenario menggunakan metrik evaluasi regresi seperti Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), dan R-squared (R²) untuk memilih model dengan performa terbaik sebagai solusi prediksi emisi CO2 untuk masing-masing interval.
## <a name="_vagjpc5645s0"></a>**Data Understanding**
Dataset yang digunakan dalam proyek ini terdiri dari dua sumber utama yang kemudian digabungkan: data historis emisi CO2 dan data cuaca dari area perkebunan kelapa sawit. Data CO2 dikumpulkan dari beberapa file CSV yang kemudian digabungkan, sedangkan data cuaca juga berasal dari file CSV terpisah. Kedua dataset ini kemudian digabungkan berdasarkan kolom waktu. Setelah proses penggabungan dan pembersihan awal (termasuk resampling per menit), dataset yang digunakan untuk pemodelan memiliki **20160 sampel data** (sebelum penghapusan baris dengan NaN akibat pembuatan fitur lag dan target).

**Sumber Data:**

- Data CO2: Kumpulan file CSV dari folder dataset/data-raw/co2.
- Data Cuaca: File CSV dari folder dataset/data-raw/cuaca.
- Data Gabungan Awal: dataset/data-clean/data\_output\_collecting.csv (hasil dari notebook data-load-merge.ipynb).
- Data Setelah EDA & Preprocessing Awal: dataset/data-clean/data\_eda\_step1.csv (hasil dari notebook eda-prepo.ipynb).
- Data Siap Model (setelah imputasi & scaling): dataset/data-clean/data\_scaled.csv (hasil dari notebook eda-prepo.ipynb).
### <a name="_43hpnbscb19m"></a>**Variabel-variabel pada dataset gabungan (data\_scaled.csv) adalah sebagai berikut:**
- **timestamp**: Waktu pencatatan data (telah diresample per menit dan dijadikan index pada tahap modeling). Tipe data: Datetime.
- **co2**: Konsentrasi karbon dioksida (variabel target, telah dinormalisasi). Satuan: ppm (setelah normalisasi, nilainya antara 0-1). Tipe data: Numerik (float64).
- **temperature**: Temperatur udara (telah dinormalisasi). Satuan: Celcius (°C) (setelah normalisasi, nilainya antara 0-1). Tipe data: Numerik (float64).
- **humidity**: Kelembaban relatif udara (telah dinormalisasi). Satuan: Persentase (%) (setelah normalisasi, nilainya antara 0-1). Tipe data: Numerik (float64).
- **rainfall**: Jumlah curah hujan (telah dinormalisasi). Satuan: milimeter (mm) (setelah normalisasi, nilainya antara 0-1). Tipe data: Numerik (float64).
- **pyrano**: Radiasi matahari (diukur oleh pyranometer, telah dinormalisasi). Satuan: Watt per meter persegi (W/m²) (setelah normalisasi, nilainya antara 0-1). Tipe data: Numerik (float64).
- **hour**: Jam dalam sehari saat pencatatan data (hasil ekstraksi dari timestamp). Tipe data: Numerik (int64, 0-23).
- **date**: Tanggal pencatatan data (hasil ekstraksi dari timestamp). Tipe data: Object (string).
### <a name="_ndelb6g1noie"></a>**Exploratory Data Analysis (EDA)**
Pada tahap EDA (dilakukan di notebook eda-prepo.ipynb), beberapa temuan penting meliputi:

- **Struktur dan Tipe Data:** Dataset awal (data\_output\_collecting.csv) terdiri dari 20.160 baris dan 6 kolom. Kolom timestamp awalnya bertipe object dan kemudian dikonversi menjadi datetime.
- **Analisis Missing Value:**
  - Kolom co2 memiliki missing values yang signifikan (sekitar 12428 dari 20160 baris, atau ~61.6%).
  - Kolom temperature, humidity, rainfall, dan pyrano juga memiliki missing values (sekitar 2406 baris, atau ~11.9%).
  - Visualisasi heatmap missing values membantu mengidentifikasi pola data yang hilang.
- **Statistik Deskriptif:**
  - co2: Mean sekitar 458.71 ppm, dengan variasi yang cukup besar (std dev 78.08).
  - temperature: Rata-rata 28.24 °C, dengan nilai maksimum mencapai 50.99 °C yang diidentifikasi sebagai potensi outlier.
  - humidity: Rata-rata 87.24%, dengan sebagian besar data di atas 70%.
  - rainfall: Mayoritas bernilai 0, mengindikasikan banyak waktu tidak terjadi hujan.
  - pyrano: Distribusi sangat condong ke kanan, dengan banyak nilai nol (malam hari atau mendung).
- **Distribusi dan Outlier:** Visualisasi histogram dan boxplot menunjukkan adanya outlier pada beberapa fitur numerik, terutama temperature, co2, dan pyrano.
- **Korelasi Antar Variabel:** Matriks korelasi (heatmap) digunakan untuk melihat hubungan linear antar variabel.
- **Time-Series Plot:** Tren semua fitur terhadap waktu diplot untuk mengevaluasi pola musiman dan fluktuasi.
- **Resampling per Jam dan Pola Harian:** Data diringkas per jam dan rata-rata per jam dihitung untuk mengidentifikasi pola harian.
## <a name="_hpxxujvv8h8p"></a>**Data Preparation**
Tahapan persiapan data dilakukan secara berurutan untuk menghasilkan dataset yang siap digunakan untuk pemodelan:

1. **Penggabungan Data (Data Merging) (data-load-merge.ipynb):**
   1. **Proses:**
      1. Semua file CSV CO2 dari dataset/data-raw/co2 digabungkan. Kolom timestamp dikonversi ke datetime dan co2 ke numerik.
      1. Data CO2 di-sampling per menit menggunakan **modus** (nilai yang paling sering muncul) untuk setiap interval menit. Rentang waktu penuh per menit dibuat untuk memastikan tidak ada menit yang hilang. Hasilnya disimpan sebagai dataset/data-clean/co2\_per\_menit.csv.
      1. File CSV data cuaca dari dataset/data-raw/cuaca dibaca. Kolom timestamp dikonversi ke datetime, dan fitur cuaca ke numerik.
      1. Data cuaca di-sampling per menit menggunakan **rata-rata** untuk setiap interval menit. Rentang waktu penuh juga dibuat. Hasilnya disimpan sebagai dataset/data-clean/cuaca\_per\_menit.csv.
      1. Data CO2 per menit dan data cuaca per menit digabungkan berdasarkan kolom minute (timestamp). Hasilnya disimpan sebagai dataset/data-clean/data\_output\_collecting.csv.
   1. **Alasan:** Mengkonsolidasikan data dari berbagai sumber menjadi satu dataset utama dan menyamakan frekuensi data ke interval per menit.
1. **Konversi Tipe Data dan Pembuatan Fitur Waktu (eda-prepo.ipynb):**
   1. **Proses:**
      1. Kolom timestamp pada data\_output\_collecting.csv dikonversi ke format datetime.
      1. Fitur hour (jam dalam sehari) dan date diekstrak dari kolom timestamp.
   1. **Alasan:** Tipe data datetime penting untuk analisis berbasis waktu dan ekstraksi fitur waktu dapat membantu model menangkap pola temporal.
1. **Penanganan Missing Values (eda-prepo.ipynb):**
   1. **Proses:**
      1. Untuk kolom co2, missing values diisi menggunakan metode **interpolasi linear** (df['co2'].interpolate(method='linear', inplace=True)).
      1. Untuk fitur cuaca (temperature, humidity, rainfall, pyrano), missing values diisi menggunakan metode **forward fill** (fillna(method='ffill')) diikuti **backward fill** (fillna(method='bfill')) untuk menangani NaN di awal atau akhir data.
      1. Hasilnya disimpan sebagai dataset/data-clean/data\_filled.csv.
   1. **Alasan:** Mengisi nilai yang hilang untuk memaksimalkan jumlah data yang dapat digunakan dan mencegah error pada tahap pemodelan. Interpolasi linear cocok untuk data time series seperti CO2, sedangkan ffill/bfill efektif untuk data cuaca yang cenderung persisten.
1. **Feature Scaling (Normalization) (eda-prepo.ipynb):**
   1. **Proses:** Semua fitur numerik yang akan digunakan dalam pemodelan (co2, temperature, humidity, rainfall, pyrano) dinormalisasi menggunakan MinMaxScaler dari Scikit-learn. Ini mengubah skala fitur sehingga nilainya berada di antara 0 dan 1.
   1. **Alasan:** Normalisasi penting untuk algoritma yang sensitif terhadap skala fitur, seperti SVR, dan dapat membantu konvergensi model lebih cepat.
1. **Pembuatan Fitur Lag dan Target (model-ml.ipynb):**
   1. **Proses:**
      1. Dibuat kolom target untuk prediksi 1 menit ke depan (co2\_target\_1min) dan 1 jam ke depan (co2\_target\_1hour) dengan menggeser (shift) kolom co2.
      1. Dibuat 10 fitur lag (co2\_lag\_1 hingga co2\_lag\_10) dari nilai co2 sebelumnya.
      1. Baris dengan nilai NaN yang muncul akibat operasi shift dihapus.
   1. **Alasan:** Fitur lag membantu model memahami dependensi temporal dalam data CO2. Target yang digeser memungkinkan prediksi nilai di masa depan.
1. **Pembagian Data (Train-Test Split) (model-ml.ipynb):**
   1. **Proses:** Dataset untuk prediksi 1 menit dan 1 jam masing-masing dibagi menjadi data latih (80%) dan data uji (20%). Pembagian dilakukan tanpa pengacakan (shuffle=False) untuk menjaga urutan waktu, yang krusial untuk data time series.
   1. **Alasan:** Untuk melatih model pada satu set data dan mengevaluasinya pada set data lain yang tidak terlihat sebelumnya, guna mendapatkan estimasi performa yang objektif.
## <a name="_ieenxz7fd3tv"></a>**Modeling**
Pada tahap ini, dilakukan pengembangan model machine learning untuk memprediksi emisi CO2. Tiga algoritma regresi dipilih untuk proyek ini: Support Vector Regression (SVR), Random Forest Regressor, dan XGBoost Regressor. Proses ini dilakukan untuk dua skenario: prediksi dengan interval 1 menit dan prediksi dengan interval 1 jam. Parameter default dari library Scikit-learn dan XGBoost digunakan untuk pelatihan awal.
### <a name="_royud9ruzccl"></a>**1. Support Vector Regression (SVR)**
- **Parameter Utama yang Digunakan (Contoh):** kernel (default 'rbf'), C (default 1.0), epsilon (default 0.1).
- **Kelebihan SVR:** Efektif di ruang dimensi tinggi, hemat memori.
- **Kekurangan SVR:** Kurang efisien untuk sampel besar, performa bergantung pada parameter, kurang interpretatif.

\# Contoh snippet kode inisialisasi SVR

from sklearn.svm import SVR

svr\_model = SVR() # Menggunakan parameter default

\# svr\_model.fit(X\_train\_scaled, y\_train)

### <a name="_kkzahq5ilqbz"></a>**2. Random Forest Regressor**
- **Parameter Utama yang Digunakan (Contoh):** n\_estimators (default 100), max\_depth (default None), random\_state=42.
- **Kelebihan Random Forest:** Efektif, tidak mudah overfitting, menangani banyak fitur, memberikan feature importance.
- **Kekurangan Random Forest:** Lambat jika pohon banyak, kurang interpretatif.

\# Contoh snippet kode inisialisasi Random Forest

from sklearn.ensemble import RandomForestRegressor

rf\_model = RandomForestRegressor(random\_state=42) # Menggunakan parameter default dengan random\_state

\# rf\_model.fit(X\_train, y\_train)

### <a name="_s0t64ryba90k"></a>**3. XGBoost Regressor**
- **Parameter Utama yang Digunakan (Contoh):** objective='reg:squarederror', n\_estimators (default 100), learning\_rate (default 0.3), max\_depth (default 6), random\_state=42.
- **Kelebihan XGBoost:** Performa prediksi yang sangat baik, cepat, menangani missing values, regularisasi.
- **Kekurangan XGBoost:** Banyak hyperparameter yang perlu di-tuning, bisa jadi kompleks.

\# Contoh snippet kode inisialisasi XGBoost

import xgboost as xgb

xgb\_model = xgb.XGBRegressor(objective='reg:squarederror', random\_state=42) # Menggunakan parameter default dengan random\_state

\# xgb\_model.fit(X\_train, y\_train)

### <a name="_ggsie6r0nxe2"></a>**Proses Improvement Model / Pemilihan Model Terbaik**
- **Hyperparameter Tuning:**
  - [Pada proyek ini, hyperparameter tuning tidak secara eksplisit dilakukan dalam notebook yang disediakan, model dilatih dengan parameter default. Jika dilakukan, jelaskan prosesnya di sini.]
- **Pemilihan Model Terbaik:**
  - Model terbaik dipilih berdasarkan skor komposit yang memperhitungkan peringkat model pada setiap metrik evaluasi (MAE, MSE, RMSE, R2) pada data uji untuk masing-masing skenario prediksi.
  - **Untuk Prediksi 1 Menit:** Random Forest Regressor dipilih sebagai model terbaik dengan skor komposit tertinggi (12.0).
  - **Untuk Prediksi 1 Jam:** SVM Regressor dipilih sebagai model terbaik dengan skor komposit tertinggi (12.0).
  - Model terbaik untuk setiap horizon kemudian dilatih ulang menggunakan keseluruhan data yang tersedia (X\_1min\_all, y\_1min\_all dan X\_1hour\_all, y\_1hour\_all) dan disimpan menggunakan joblib ke folder models/.
## <a name="_s6m4mueeitsn"></a>**Evaluation**
Untuk mengevaluasi performa model-model regresi yang telah dibangun, beberapa metrik evaluasi standar untuk masalah regresi digunakan. Metrik-metrik ini membantu mengukur seberapa dekat prediksi model dengan nilai aktual emisi CO2.

Metrik evaluasi yang digunakan adalah:

1. **Mean Absolute Error (MAE)**
   1. **Formula:** MAE = (1/n) \* Σ|y\_i - ŷ\_i|
   1. **Penjelasan:** MAE mengukur rata-rata selisih absolut antara nilai aktual dan nilai prediksi. MAE memberikan gambaran seberapa besar error prediksi secara rata-rata, dalam satuan yang sama dengan variabel target. Nilai MAE yang lebih rendah menunjukkan performa model yang lebih baik. Kelebihannya adalah mudah diinterpretasikan dan tidak terlalu sensitif terhadap outliers dibandingkan MSE.
1. **Mean Squared Error (MSE)**
   1. **Formula:** MSE = (1/n) \* Σ(y\_i - ŷ\_i)²
   1. **Penjelasan:** MSE mengukur rata-rata dari kuadrat selisih antara nilai aktual dan nilai prediksi. Dengan mengkuadratkan error, MSE memberikan bobot yang lebih besar pada error yang besar (outliers). Satuan MSE adalah kuadrat dari satuan variabel target, sehingga terkadang kurang intuitif. Nilai MSE yang lebih rendah menunjukkan performa model yang lebih baik.
1. **Root Mean Squared Error (RMSE)**
   1. **Formula:** RMSE = sqrt(MSE) = sqrt((1/n) \* Σ(y\_i - ŷ\_i)²)
   1. **Penjelasan:** RMSE adalah akar kuadrat dari MSE. Keuntungannya adalah RMSE memiliki satuan yang sama dengan variabel target, sehingga lebih mudah diinterpretasikan dibandingkan MSE. Seperti MSE, RMSE juga memberikan bobot lebih pada error yang besar. Nilai RMSE yang lebih rendah menunjukkan performa model yang lebih baik dan merupakan salah satu metrik yang paling umum digunakan untuk mengevaluasi model regresi.
1. **R-squared (R² atau Koefisien Determinasi)**
   1. **Formula:** R² = 1 - (SS\_res / SS\_tot)
      1. SS\_res (Sum of Squares of Residuals): Σ(y\_i - ŷ\_i)²
      1. SS\_tot (Total Sum of Squares): Σ(y\_i - ȳ)², di mana ȳ adalah mean dari nilai aktual.
   1. **Penjelasan:** R-squared mengukur proporsi varians dalam variabel dependen (target) yang dapat diprediksi dari variabel independen (fitur). Nilai R² berkisar antara 0 dan 1 (atau bisa negatif jika model sangat buruk). Nilai R² yang lebih tinggi (mendekati 1) menunjukkan bahwa model lebih baik dalam menjelaskan variabilitas data. R² = 0.75 berarti model dapat menjelaskan 75% varians pada data target.
### <a name="_bci9h8uj1854"></a>**Hasil Proyek Berdasarkan Metrik Evaluasi**
#### <a name="_5ghujlroo6gq"></a>**Prediksi Interval 1 Menit (Notebook)**
Berikut adalah tabel hasil evaluasi dari ketiga model pada data uji untuk prediksi 1 menit:

|**Model**|**MAE**|**MSE**|**RMSE**|**R-squared (R²)**|
| :-: | :-: | :-: | :-: | :-: |
|SVM Regressor|0\.050948|0\.004749|0\.068915|0\.884463|
|Random Forest|0\.022472|0\.001450|0\.038077|0\.964729|
|XGBoost|0\.027126|0\.002743|0\.052376|0\.933264|

**Analisis Hasil (1 Menit):**

- Untuk prediksi dengan interval 1 menit, model **Random Forest** menunjukkan performa terbaik di semua metrik evaluasi. Model ini memiliki MAE terendah (0.022472), MSE terendah (0.001450), RMSE terendah (0.038077), dan R-squared tertinggi (0.964729).
- Hasil ini mengindikasikan bahwa Random Forest paling akurat dalam memprediksi emisi CO2 untuk jangka pendek (1 menit) dan mampu menjelaskan sekitar 96.47% varians dalam data target.
- SVM Regressor menunjukkan performa terburuk pada skenario ini, meskipun masih memiliki R-squared yang cukup baik (0.884463).
- Visualisasi perbandingan nilai aktual vs. prediksi untuk setiap model juga dibuat untuk melihat seberapa baik model mengikuti pola data aktual.
#### <a name="_rjjr01fztg5c"></a>**Prediksi Interval 1 Menit (Skrip Python)**
Berikut adalah tabel hasil evaluasi dari ketiga model pada data uji untuk prediksi 1 menit:

|**Model**|**MAE**|**MSE**|**RMSE**|**R-squared (R²)**|
| :-: | :-: | :-: | :-: | :-: |
|SVM Regressor|0\.039986|0\.002573|0\.050728|0\.937398|
|Random Forest|0\.023009|0\.001649|0\.040610|0\.959879|
|**XGBoost**|**0.017958**|**0.001057**|**0.032519**|**0.974274**|

**Analisis Hasil (1 Menit):**

- Untuk prediksi dengan interval 1 menit, model **XGBoost** menunjukkan performa terbaik di semua metrik evaluasi. Model ini memiliki MAE terendah (0.017958), MSE terendah (0.001057), RMSE terendah (0.032519), dan R-squared tertinggi (0.974274).
- Hasil ini mengindikasikan bahwa XGBoost paling akurat dalam memprediksi emisi CO2 untuk jangka pendek (1 menit) dan mampu menjelaskan sekitar 97.43% varians dalam data target.
- Random Forest juga menunjukkan performa yang sangat baik, sementara SVM Regressor memiliki performa sedikit di bawahnya namun tetap dengan R-squared yang tinggi.
- Visualisasi perbandingan nilai aktual vs. prediksi untuk setiap model (plot aktual vs prediksi) dikomentari pada file main.py (skrip Python) untuk mempercepat proses eksekusi. Namun, idealnya divisualisasikan untuk analisis lebih lanjut.
#### <a name="_sbnfkynjo77s"></a>**Prediksi Interval 1 Jam (Notebook)**
Berikut adalah tabel hasil evaluasi dari ketiga model pada data uji untuk prediksi 1 jam:

|**Model**|**MAE**|**MSE**|**RMSE**|**R-squared (R²)**|
| :-: | :-: | :-: | :-: | :-: |
|SVM Regressor|0\.079888|0\.011128|0\.105489|0\.729485|
|Random Forest|0\.089402|0\.013316|0\.115394|0\.676302|
|XGBoost|0\.095086|0\.014940|0\.122231|0\.636805|

**Analisis Hasil (1 Jam):**

- Untuk prediksi dengan interval 1 jam, model **SVM Regressor** menunjukkan performa terbaik di semua metrik evaluasi. Model ini memiliki MAE terendah (0.079888), MSE terendah (0.011128), RMSE terendah (0.105489), dan R-squared tertinggi (0.729485).
- Ini menunjukkan bahwa SVM Regressor lebih unggul dalam memprediksi emisi CO2 untuk jangka waktu yang lebih panjang (1 jam) dibandingkan dua model lainnya, dengan kemampuan menjelaskan sekitar 72.95% varians data.
- Pada skenario ini, XGBoost menunjukkan performa terendah. Perlu dicatat bahwa performa semua model menurun (R-squared lebih rendah) untuk prediksi 1 jam dibandingkan prediksi 1 menit, yang mengindikasikan tantangan yang lebih besar dalam memprediksi untuk horizon waktu yang lebih jauh.
- Visualisasi perbandingan nilai aktual vs. prediksi juga dibuat untuk skenario ini.
#### <a name="_l00s464suah5"></a>**Prediksi Interval 1 Jam (Skrip Python)**
Berikut adalah tabel hasil evaluasi dari ketiga model pada data uji untuk prediksi 1 jam:

|**Model**|**MAE**|**MSE**|**RMSE**|**R-squared (R²)**|
| :-: | :-: | :-: | :-: | :-: |
|**SVM Regressor**|**0.058026**|**0.005581**|**0.074703**|**0.864339**|
|Random Forest|0\.070744|0\.008339|0\.091317|0\.797286|
|XGBoost|0\.071561|0\.008726|0\.093410|0\.787888|

**Analisis Hasil (1 Jam):**

- Untuk prediksi dengan interval 1 jam, model **SVM Regressor** menunjukkan performa terbaik di semua metrik evaluasi. Model ini memiliki MAE terendah (0.058026), MSE terendah (0.005581), RMSE terendah (0.074703), dan R-squared tertinggi (0.864339).
- Ini menunjukkan bahwa SVM Regressor lebih unggul dalam memprediksi emisi CO2 untuk jangka waktu yang lebih panjang (1 jam) dibandingkan dua model lainnya, dengan kemampuan menjelaskan sekitar 86.43% varians data.
- Perlu dicatat bahwa performa semua model menurun (R-squared lebih rendah) untuk prediksi 1 jam dibandingkan prediksi 1 menit, yang mengindikasikan tantangan yang lebih besar dalam memprediksi untuk horizon waktu yang lebih jauh. Namun, R-squared untuk SVM pada 1 jam (0.864339) masih menunjukkan performa yang baik.
- Visualisasi perbandingan nilai aktual vs. prediksi untuk setiap model (plot aktual vs prediksi) dikomentari pada file main.py (skrip Python) untuk mempercepat proses eksekusi.

**Ringkasan Pemilihan Model Berdasarkan Skor Komposit (Berdasarkan Notebook):** Berdasarkan analisis skor komposit yang memperhitungkan semua metrik:

- Model **Random Forest** (Skor: 12.0) dipilih sebagai model final untuk prediksi interval 1 menit.
- Model **SVM Regressor** (Skor: 12.0) dipilih sebagai model final untuk prediksi interval 1 jam.

**Ringkasan Pemilihan Model Berdasarkan Skor Komposit (Berdasarkan Skrip Python):** Berdasarkan analisis skor komposit yang memperhitungkan semua metrik:

- Model **XGBoost** (Skor: 12.0) dipilih sebagai model final untuk prediksi interval 1 menit.
- Model **SVM Regressor** (Skor: 12.0) dipilih sebagai model final untuk prediksi interval 1 jam.

Disini terdapat perbedaan hasil antara yang di notebook dan yg di python pada prediksi 1 menit, yang belum saya ketahui apa yg membedakannya.

Temuan ini menarik karena model terbaik berbeda tergantung pada horizon waktu prediksi. Ini menunjukkan bahwa karakteristik data dan kemampuan model dalam menangkap pola jangka pendek dan jangka panjang dapat bervariasi.

