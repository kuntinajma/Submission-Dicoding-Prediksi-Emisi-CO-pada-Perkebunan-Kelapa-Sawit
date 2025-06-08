# **Laporan Proyek Machine Learning \- Kunti Najma Jalia**

## **Domain Proyek \- Prediksi Emisi CO₂ pada Perkebunan Kelapa Sawit**

Perkebunan kelapa sawit (*Elaeis guineensis Jacq*.) memainkan peran penting dalam perekonomian Indonesia, menjadikan negara ini sebagai salah satu produsen dan pengekspor minyak sawit mentah (CPO) terbesar di dunia (Arjuna & Santosa, 2018). Selain menyumbang devisa, sektor ini juga menjadi sumber penghidupan bagi jutaan masyarakat. Namun demikian, industri kelapa sawit kerap dikritik karena dampak lingkungannya, khususnya terkait emisi gas rumah kaca (GRK) seperti karbon dioksida (CO₂), yang berkontribusi besar terhadap pemanasan global (Hong, 2023). Emisi ini kerap digunakan sebagai alasan dalam kampanye negatif terhadap sawit di pasar global (Maisarah et al., 2024).

Sumber utama emisi CO₂ dalam perkebunan sawit berasal dari konversi lahan, terutama saat hutan atau lahan gambut diubah menjadi areal perkebunan (Putri et al., 2016). Proses ini menyebabkan hilangnya cadangan karbon dalam biomassa dan tanah, serta oksidasi gambut yang melepaskan CO₂ dalam jumlah besar (Arjuna & Santosa, 2018). Selain itu, penggunaan pupuk nitrogen berkontribusi terhadap pelepasan N₂O, gas rumah kaca yang jauh lebih kuat dibanding CO₂ (K. et al., 2024). Aktivitas lainnya seperti respirasi tanah, penggunaan bahan bakar untuk alat berat dan transportasi, serta pengolahan limbah cair pabrik sawit (POME), juga menyumbang pada peningkatan emisi karbon (Arjuna & Santosa, 2018).

Sebagai solusi, pendekatan berbasis teknologi menjadi sangat relevan. *Precision Agriculture Technologies* (PAT) berpotensi mengoptimalkan input pertanian untuk menekan emisi (K. et al., 2024), sementara *machine learning* (ML) dapat digunakan untuk membangun model prediktif berbasis data historis kondisi lingkungan dan cuaca (Bussaban et al., 2024). Dengan model prediksi yang akurat, pengelola dapat merancang strategi berbasis data untuk mengidentifikasi sumber emisi, meningkatkan efisiensi sumber daya, serta menyesuaikan praktik budidaya agar lebih berkelanjutan. Hal ini juga mendukung pelaporan emisi yang lebih transparan sesuai dengan standar nasional dan global (Arjuna & Santosa, 2018), serta meningkatkan daya saing kelapa sawit Indonesia di pasar internasional yang semakin peduli terhadap isu iklim.

**Referensi:**

* Arjuna, R., & Santosa, E. (2018). Asesmen carbon footprint pada produksi minyak kelapa sawit (*Elaeis guineensis* Jacq.) di Kebun Sei Lukut, Kabupaten Siak, Riau. *Buletin Agrohorti*. 6(2): 287–295.  
* Bussaban, K., Kularbphettong, K., Raksuntorn, N., & Boonseng, C. (2024). Prediction of CO₂ emissions using machine learning. *Edelweiss Applied Science and Technology*. 8(4): 1–11.  
* Hong, W. O. (2023). Review on carbon footprint of the palm oil industry: Insights into recent developments. *International Journal of Sustainable Development and Planning*. 18(2): 447–455.  
* K., F., Betül, R. S., & B., A. V. (2024). The importance and contribution of precision agriculture technologies in reducing greenhouse gas emissions. *International Journal of Advanced Research*. 12(12): 944–959.  
* Maisarah, Dian, R., & Febrianto, E. B. (2024). Uji serapan karbon (sekuestrasi) tanaman kelapa sawit sebagai baseline pada trading karbon. *Ranah Research: Journal of Multidisciplinary Research and Development*. 7(1): 633–643.  
* Putri, T. T. A., Syaufina, L., & Anshari, G. Z. (2016). Emisi karbon dioksida (CO₂) rizosfer dan non rizosfer dari perkebunan kelapa sawit (*Elaeis guineensis*) pada lahan gambut dangkal. *Jurnal Tanah dan Iklim*. 40(1): 43–50.

## **Business Understanding**

Proses klarifikasi masalah dalam proyek ini melibatkan pemahaman mendalam terhadap kebutuhan untuk memprediksi emisi CO2 di perkebunan kelapa sawit dan bagaimana machine learning dapat memberikan solusi yang bernilai.

### **Problem Statements**

1. Bagaimana cara memprediksi secara akurat jumlah emisi gas CO2 yang dihasilkan oleh perkebunan kelapa sawit berdasarkan data historis kondisi lingkungan dan cuaca untuk interval waktu 1 menit dan 1 jam ke depan?  
2. Algoritma machine learning manakah (antara Support Vector Machine, Random Forest, dan XGBoost) yang memberikan performa terbaik dalam memprediksi emisi CO2 untuk interval prediksi 1 menit dan 1 jam?

### **Goals**

1. Mengembangkan model regresi machine learning yang mampu memprediksi tingkat emisi CO2 dengan tingkat akurasi yang dapat diterima (berdasarkan beberapa matriks evaluasi) untuk prediksi 1 menit dan 1 jam.   
2. Mengevaluasi dan membandingkan kinerja model Support Vector Regression (SVR), Random Forest Regressor, dan XGBoost Regressor untuk menentukan model prediksi CO2 yang paling optimal untuk setiap interval prediksi 1 menit dan 1 jam.

### **Solution statements**

Untuk mencapai tujuan-tujuan di atas, tahapan solusi yang akan dilakukan dalam proyek ini adalah sebagai berikut:

1. **Pengumpulan Data**  
   Data yang digunakan meliputi data konsentrasi CO₂ dan data cuaca (suhu, kelembapan, kecepatan angin, curah hujan) dari sensor yang dipasang di perkebunan kelapa sawit.  
2. **Pemrosesan Awal Data (Data Preprocessing)**  
- Menggabungkan dataset CO₂ dan cuaca berdasarkan stempel waktu (*timestamp*).  
- Menangani nilai yang hilang (*missing values*) menggunakan metode yang sesuai (misalnya, imputasi berdasarkan statistik atau metode pengisian maju/mundur).  
- Melakukan normalisasi atau standarisasi fitur jika diperlukan oleh algoritma *machine learning* tertentu.  
3. **Analisis Data Eksploratif (Exploratory Data Analysis \- EDA)**  
- Melakukan analisis statistik deskriptif untuk memahami distribusi masing-masing variabel.  
- Memvisualisasikan data untuk mengidentifikasi pola, tren, *outlier*, dan hubungan antar variabel (misalnya, menggunakan histogram, *boxplot*, *scatterplot*, dan matriks korelasi).  
- Menganalisis korelasi antara variabel iklim dengan konsentrasi CO₂.  
4. **Persiapan Data untuk Pemodelan (Data Preparation for Modeling)**  
- Membuat fitur target (*target variable*), yaitu konsentrasi CO₂ yang akan diprediksi untuk interval waktu tertentu (1 menit dan 1 jam ke depan).  
- Memilih fitur-fitur (*feature selection*) yang akan digunakan sebagai input model.  
- Membagi dataset menjadi data latih (*training set*) dan data uji (*testing set*).  
5. **Pengembangan Model (Model Development)**  
- Melatih beberapa algoritma *machine learning* untuk tugas regresi, antara lain:  
1. Support Vector Regression (SVR)  
2. Random Forest Regressor  
3. XGBoost Regressor  
- Melakukan penyetelan *hyperparameter* (jika diperlukan) untuk mendapatkan performa model yang optimal.  
6. **Evaluasi Model (Model Evaluation)**  
- Mengevaluasi performa model pada data uji menggunakan metrik evaluasi regresi yang umum, seperti:  
1. Mean Absolute Error (MAE)  
2. Mean Squared Error (MSE)  
3. Root Mean Squared Error (RMSE)  
4. R-squared (R²)  
- Membandingkan hasil dari berbagai model dan memilih model terbaik untuk setiap skenario prediksi (1 menit dan 1 jam).  
7. **Penyimpanan** dan Kesimpulan Model (Model Saving and **Conclusion):**  
- Menyimpan model terbaik yang telah dilatih sehingga dapat digunakan kembali di masa mendatang.  
- Menarik kesimpulan berdasarkan hasil analisis dan evaluasi model, serta memberikan rekomendasi untuk implementasi praktis.

Pendekatan ini diharapkan dapat menghasilkan model prediksi CO₂ yang akurat, serta memberikan pemahaman yang lebih mendalam mengenai faktor-faktor yang mempengaruhi emisi CO₂ di perkebunan kelapa sawit.

## **Data Understanding**

Dataset yang digunakan dalam proyek ini terdiri dari dua sumber utama yang kemudian digabungkan: data historis emisi CO2 dan data cuaca dari area perkebunan kelapa sawit. Data CO2 dikumpulkan dari beberapa file CSV yang kemudian digabungkan, sedangkan data cuaca juga berasal dari file CSV terpisah. Kedua dataset ini kemudian digabungkan berdasarkan kolom waktu. Setelah proses penggabungan dan pembersihan awal (termasuk resampling per menit), dataset yang digunakan untuk pemodelan memiliki **20160 sampel data** (sebelum penghapusan baris dengan NaN akibat pembuatan fitur lag dan target).

**Sumber Data:**

* Data CO2 dan Cuaca: Dataset diperoleh dari sensor IoT dari Eddy Covariance yang kemudian saya upload secara publik melalui platform Kaggle yang dapat diakses melalui link berikut [https://www.kaggle.com/datasets/kuntinajmajalia/emisi-karbon-perkebunan-kelapa-sawit/data](https://www.kaggle.com/datasets/kuntinajmajalia/emisi-karbon-perkebunan-kelapa-sawit/data)   
* Data Gabungan Awal: `dataset/data-clean/data_output_collecting.csv`
* Data Setelah EDA & Preprocessing Awal: `dataset/data-clean/data_eda_step1.csv`
* Data Siap Model (setelah imputasi & scaling): `dataset/data-clean/data_scaled.csv`

### **Variabel-variabel pada dataset gabungan (`data_scaled.csv`) adalah sebagai berikut:**

* **timestamp**: Waktu pencatatan data (telah diresample per menit dan dijadikan index pada tahap modeling). Tipe data: Datetime.  
* **co2**: Konsentrasi karbon dioksida (variabel target, telah dinormalisasi). Satuan: ppm (setelah normalisasi, nilainya antara 0-1). Tipe data: Numerik (float64).  
* **temperature**: Temperatur udara (telah dinormalisasi). Satuan: Celcius (°C) (setelah normalisasi, nilainya antara 0-1). Tipe data: Numerik (float64).  
* **humidity**: Kelembaban relatif udara (telah dinormalisasi). Satuan: Persentase (%) (setelah normalisasi, nilainya antara 0-1). Tipe data: Numerik (float64).  
* **rainfall**: Jumlah curah hujan (telah dinormalisasi). Satuan: milimeter (mm) (setelah normalisasi, nilainya antara 0-1). Tipe data: Numerik (float64).  
* **pyrano**: Radiasi matahari (diukur oleh pyranometer, telah dinormalisasi). Satuan: Watt per meter persegi (W/m²) (setelah normalisasi, nilainya antara 0-1). Tipe data: Numerik (float64).  

### **Exploratory Data Analysis (EDA)**

Pada tahap EDA, beberapa temuan penting meliputi:

* **Struktur dan Tipe Data:** Dataset awal (`data_output_collecting.csv`) terdiri dari 20.160 baris dan 6 kolom. Kolom `timestamp` awalnya bertipe object dan kemudian dikonversi menjadi datetime.  
* **Analisis Missing Value:**  
- Kolom `co2` memiliki missing values yang signifikan (sekitar 12428 dari 20160 baris, atau \~61.6%).  
- Kolom `temperature`, `humidity`, `rainfall`, dan `pyrano` juga memiliki missing values (sekitar 2406 baris, atau \~11.9%).  
- Visualisasi heatmap missing values membantu mengidentifikasi pola data yang hilang.  
* **Statistik Deskriptif:**  
- `co2`: Mean sekitar 458.71 ppm, dengan variasi yang cukup besar (std dev 78.08).  
- `temperature`: Rata-rata 28.24 °C, dengan nilai maksimum mencapai 50.99 °C yang diidentifikasi sebagai potensi outlier.  
- `humidity`: Rata-rata 87.24%, dengan sebagian besar data di atas 70%.  
- `rainfall`: Mayoritas bernilai 0, mengindikasikan banyak waktu tidak terjadi hujan.  
- `pyrano`: Distribusi sangat condong ke kanan, dengan banyak nilai nol (malam hari atau mendung).  
* **Distribusi dan Outlier:** Visualisasi histogram dan boxplot menunjukkan adanya outlier pada beberapa fitur numerik, terutama `temperature`, `co2`, dan `pyrano`.  
* **Korelasi Antar Variabel:** Matriks korelasi (heatmap) digunakan untuk melihat hubungan linear antar variabel.  
* **Time-Series Plot:** Tren semua fitur terhadap waktu diplot untuk mengevaluasi pola musiman dan fluktuasi.  
* **Resampling per Jam dan Pola Harian:** Data diringkas per jam dan rata-rata per jam dihitung untuk mengidentifikasi pola harian.

## **Data Preparation**

Tahapan persiapan data dilakukan secara berurutan untuk menghasilkan dataset yang siap digunakan untuk pemodelan:

1. **Penggabungan Data (Data Merging):**  
- **Proses:**  
1. Semua file CSV CO2 dari `dataset/data-raw/co2` digabungkan. Kolom `timestamp` dikonversi ke datetime dan `co2` ke numerik.  
2. Data CO2 di-sampling per menit menggunakan **modus** (nilai yang paling sering muncul) untuk setiap interval menit. Rentang waktu penuh per menit dibuat untuk memastikan tidak ada menit yang hilang. Hasilnya disimpan sebagai `dataset/data-clean/co2_per_menit.csv`.  
3. File CSV data cuaca dari `dataset/data-raw/cuaca` dibaca. Kolom `timestamp` dikonversi ke datetime, dan fitur cuaca ke numerik.  
4. Data cuaca di-sampling per menit untuk setiap interval menit (namun pada kasus diatas data cuaca sudah per-menit tinggal di filter saja) dan hasilnya disimpan sebagai `dataset/data-clean/cuaca_per_menit.csv`.  
5. Data CO2 per menit dan data cuaca per menit digabungkan berdasarkan kolom `minute` (timestamp). Hasilnya disimpan sebagai `dataset/data-clean/data_output_collecting.csv`.  
- **Alasan:** Mengkonsolidasikan data dari berbagai sumber menjadi satu dataset utama dan menyamakan frekuensi data ke interval per menit.  
2. **Konversi Tipe Data dan Pembuatan Fitur Waktu:**  
- **Proses:**  
1. Kolom `timestamp` pada `data_output_collecting.csv` dikonversi ke format datetime.   
- **Alasan:** Tipe data datetime penting untuk analisis berbasis waktu dan ekstraksi fitur waktu dapat membantu model menangkap pola temporal.  
3. **Penanganan Missing Values:**  
   Analisis pada notebook menunjukkan adanya *missing values* pada beberapa kolom, terutama pada kolom `co2` yang mencapai \~60%. Untuk mengatasi hal ini, dilakukan strategi pengisian nilai yang kosong. Metode yang diimplementasikan pada kode adalah forward fill (`ffill`) yang diikuti dengan backward fill (`bfill`) untuk semua kolom dalam DataFrame.  
* `ffill()` (Forward Fill): Metode ini mengisi nilai yang hilang dengan menggunakan nilai valid terakhir yang diketahui sebelumnya dalam urutan data. Ini cocok untuk data deret waktu, dengan asumsi bahwa sebuah kondisi (misalnya level CO₂) akan tetap sama sampai pengukuran baru tercatat.  
* `bfill()` (Backward Fill): Setelah `ffill`, jika masih ada nilai kosong di awal dataset (di mana belum ada nilai valid sebelumnya untuk melakukan `ffill`), `bfill` akan mengisinya dengan menggunakan nilai valid berikutnya dalam urutan data.  
  Implementasi di kode: `df_filled = df.ffill().bfill()`. Kombinasi ini memastikan tidak ada lagi nilai yang hilang di seluruh dataset, sehingga data siap untuk tahap pemodelan. Hasilnya disimpan di `dataset/data-clean/data_filled.csv`.  
4. **Feature Scaling (Normalization):**  
- **Proses:** Semua fitur numerik yang akan digunakan dalam pemodelan (`co2`, `temperature`, `humidity`, `rainfall`, `pyrano`) dinormalisasi menggunakan `MinMaxScaler` dari Scikit-learn. Ini mengubah skala fitur sehingga nilainya berada di antara 0 dan 1\.  
- **Alasan:** Normalisasi penting untuk algoritma yang sensitif terhadap skala fitur, seperti SVR, dan dapat membantu konvergensi model lebih cepat.  
5. **Pembuatan Fitur Lag dan Target:**  
- **Proses:**  
1. Dibuat kolom target untuk prediksi 1 menit ke depan (`co2_target_1min`) dan 1 jam ke depan (`co2_target_1hour`) dengan menggeser (`shift`) kolom `co2`.  
2. Dibuat 10 fitur lag (`co2_lag_1` hingga `co2_lag_10`) dari nilai `co2` sebelumnya.  
3. Baris dengan nilai NaN yang muncul akibat operasi `shift` dihapus.  
- **Alasan:** Fitur lag membantu model memahami dependensi temporal dalam data CO2. Target yang digeser memungkinkan prediksi nilai di masa depan.  
6. **Pembagian Data (Train-Test Split):**  
- **Proses:** Dataset untuk prediksi 1 menit dan 1 jam masing-masing dibagi menjadi data latih (80%) dan data uji (20%). Pembagian dilakukan tanpa pengacakan (`shuffle=False`) untuk menjaga urutan waktu, yang krusial untuk data time series.  
- **Alasan:** Untuk melatih model pada satu set data dan mengevaluasinya pada set data lain yang tidak terlihat sebelumnya, guna mendapatkan estimasi performa yang objektif.

## **Modeling**

Pada tahap ini, beberapa algoritma *machine* learning untuk regresi akan dilatih dan dievaluasi guna menemukan model terbaik untuk memprediksi konsentrasi CO₂. Proses ini dilakukan dalam *notebook* dan juga diimplementasikan dalam skrip `main.py`.

Model yang digunakan adalah:

1. **Support Vector Regression (SVR)**  
2. **Random Forest Regressor**  
3. **XGBoost Regressor**

Untuk setiap model, pelatihan dan evaluasi dilakukan secara terpisah untuk dua skenario target prediksi:

* Prediksi konsentrasi CO₂ untuk **1 menit ke depan**.  
* Prediksi konsentrasi CO₂ untuk **1 jam (60 menit) ke depan**.

Parameter default digunakan untuk setiap model pada iterasi awal ini sesuai *notebook* dan `main.py` yang menggunakan `n_estimators=100` untuk Random Forest dan XGBoost, dan kernel `rbf` untuk SVR.

### **1\. Support Vector Regression (SVR)**

Support Vector Regression (SVR) adalah algoritma *supervised learning* yang merupakan ekstensi dari Support Vector Machine (SVM) untuk menangani masalah regresi.

* **Kelebihan SVR:** Efektif di ruang dimensi tinggi, hemat memori.  
* **Kekurangan SVR:** Kurang efisien untuk sampel besar, performa bergantung pada parameter, kurang interpretatif.  
* **Cara Kerja Konseptual:** Prinsip dasar SVR adalah menemukan sebuah fungsi (atau *hyperplane* dalam ruang berdimensi tinggi) yang paling cocok untuk memprediksi nilai kontinu dari data input. Berbeda dengan regresi linear biasa yang mencoba meminimalkan total kuadrat kesalahan untuk semua titik data, SVR bekerja dengan mendefinisikan sebuah "pita" atau *margin* toleransi di sekitar *hyperplane* prediksi. Pita ini dikenal sebagai *epsilon-insensitive tube* (ε-tube). SVR hanya akan memberikan penalti atau memperhitungkan kesalahan dari titik-titik data yang berada **di luar** pita toleransi ini. Titik data yang berada di dalam pita tidak berkontribusi pada *loss function*. Tujuan SVR adalah untuk memuat sebanyak mungkin titik data ke dalam ε-tube ini sambil menjaga *flatness* dari fungsi prediksi (yaitu, menjaga agar bobot parameter model tetap kecil untuk menghindari *overfitting*). Untuk menangani hubungan data yang bersifat non-linear, SVR memanfaatkan *kernel trick* (seperti kernel RBF \- *Radial Basis Function*, polinomial, atau sigmoid) yang memetakan data input ke ruang fitur berdimensi lebih tinggi di mana pemisah linear (atau *hyperplane* regresi) dapat ditemukan dengan lebih mudah.

### **2\. Random Forest Regressor**

Random Forest adalah salah satu metode *ensemble learning* yang sangat populer dan efektif, baik untuk tugas klasifikasi maupun regresi. Untuk regresi, ia dikenal sebagai Random Forest Regressor.

* **Kelebihan Random Forest:** Efektif, tidak mudah overfitting, menangani banyak fitur, memberikan feature importance.  
* **Kekurangan Random Forest:** Lambat jika pohon banyak, kurang interpretatif.  
* **Cara Kerja Konseptual:** Algoritma ini bekerja dengan membangun sejumlah besar **pohon keputusan (*decision trees*)** secara independen selama proses pelatihan. Setiap pohon dalam "hutan" ini dibangun dari sampel data yang berbeda yang diambil secara acak dengan penggantian dari dataset asli (teknik ini disebut *bagging* atau *bootstrap aggregating*). Selain itu, pada setiap simpul (titik pemisahan) dalam pohon, bukan semua fitur yang dipertimbangkan, melainkan hanya sebagian kecil fitur yang dipilih secara acak (*feature randomness* atau *random subspace method*). Hal ini bertujuan untuk mengurangi korelasi antar pohon dan membuat masing-masing pohon lebih beragam. Untuk tugas regresi, ketika melakukan prediksi pada data baru, setiap pohon dalam Random Forest akan menghasilkan nilai prediksi. Hasil prediksi akhir dari keseluruhan Random Forest bukanlah hasil dari satu pohon tunggal, melainkan **rata-rata (mean)** dari prediksi yang dihasilkan oleh seluruh pohon individu yang telah dibangun. Dengan menggabungkan prediksi dari banyak pohon yang beragam dan cenderung memiliki *variance* tinggi namun *bias* rendah, Random Forest mampu mengurangi *variance* secara keseluruhan dan menghasilkan model yang lebih stabil, akurat, dan kurang rentan terhadap *overfitting* dibandingkan dengan satu pohon keputusan tunggal.

### **3\. XGBoost Regressor**

XGBoost (Extreme Gradient Boosting) adalah implementasi dari algoritma *gradient boosting* yang dioptimalkan untuk kecepatan, performa, dan fleksibilitas. Ini adalah salah satu algoritma yang paling sering digunakan dalam kompetisi *machine learning*.

* **Kelebihan XGBoost:** Performa prediksi yang sangat baik, cepat, menangani missing values, regularisasi.  
* **Kekurangan XGBoost:** Banyak hyperparameter yang perlu di-tuning, bisa jadi kompleks.  
* **Cara Kerja Konseptual:** XGBoost, seperti halnya algoritma *gradient boosting* lainnya, bekerja secara **sekuensial** atau iteratif dengan membangun model (biasanya pohon keputusan) satu per satu. Setiap pohon baru yang ditambahkan ke dalam *ensemble* bertujuan untuk **memperbaiki kesalahan (residu)** yang dibuat oleh kombinasi pohon-pohon sebelumnya. Proses ini disebut *boosting*. Secara lebih spesifik, algoritma ini menggunakan pendekatan *gradient descent* untuk meminimalkan *loss function* (fungsi kesalahan) ketika menambahkan pohon baru. Pohon baru "belajar" dari residu atau gradien kesalahan dari model sebelumnya. XGBoost memperkenalkan beberapa peningkatan signifikan dibandingkan *gradient boosting* tradisional, termasuk:  
1. **Regularisasi:** Menggunakan regularisasi L1 (Lasso) dan L2 (Ridge) pada bobot daun pohon untuk mencegah *overfitting* dan membuat model lebih general.  
2. **Penanganan Missing Values:** Mampu menangani nilai yang hilang secara internal dengan mempelajari arah default untuk *missing values* pada setiap simpul pohon.  
3. **Tree Pruning:** Menggunakan teknik *pruning* pohon yang lebih canggih (misalnya, berdasarkan `max_depth` dan `gamma` atau *minimum loss reduction*).  
4. **Paralelisasi dan Efisiensi:** Dioptimalkan untuk komputasi paralel dan penggunaan memori yang efisien, sehingga dapat melatih model dengan sangat cepat bahkan pada dataset besar.  
5. **Cross-validation Bawaan:** Memiliki fungsionalitas *cross-validation* yang terintegrasi.

### **Proses Improvement Model / Pemilihan Model Terbaik**

- Model terbaik dipilih berdasarkan skor komposit yang memperhitungkan peringkat model pada setiap metrik evaluasi (MAE, MSE, RMSE, R2) pada data uji untuk masing-masing skenario prediksi.  
- **Untuk Prediksi 1 Menit:** Pada Notebook Random Forest Regressor dipilih sebagai model terbaik dengan skor komposit tertinggi (12.0)  
- **Untuk Prediksi 1 Jam:** SVM Regressor dipilih sebagai model terbaik dengan skor komposit tertinggi (12.0).  
- Model terbaik untuk setiap horizon kemudian dilatih ulang menggunakan keseluruhan data yang tersedia (`X_1min_all`, `y_1min_all` dan `X_1hour_all`, `y_1hour_all`) dan disimpan menggunakan `joblib` ke folder `models/`.

## **Evaluation**

Untuk mengevaluasi performa model-model regresi yang telah dibangun, beberapa metrik evaluasi standar untuk masalah regresi digunakan. Metrik-metrik ini membantu mengukur seberapa dekat prediksi model dengan nilai aktual emisi CO2.

Metrik evaluasi yang digunakan adalah:

1. **Mean Absolute Error (MAE)**  
- **Formula:** `MAE = (1/n) * Σ|y_i - ŷ_i|`  
- **Penjelasan:** MAE mengukur rata-rata selisih absolut antara nilai aktual dan nilai prediksi. MAE memberikan gambaran seberapa besar error prediksi secara rata-rata, dalam satuan yang sama dengan variabel target. Nilai MAE yang lebih rendah menunjukkan performa model yang lebih baik. Kelebihannya adalah mudah diinterpretasikan dan tidak terlalu sensitif terhadap outliers dibandingkan MSE.  
2. **Mean Squared Error (MSE)**  
- **Formula:** `MSE = (1/n) * Σ(y_i - ŷ_i)²`  
- **Penjelasan:** MSE mengukur rata-rata dari kuadrat selisih antara nilai aktual dan nilai prediksi. Dengan mengkuadratkan error, MSE memberikan bobot yang lebih besar pada error yang besar (outliers). Satuan MSE adalah kuadrat dari satuan variabel target, sehingga terkadang kurang intuitif. Nilai MSE yang lebih rendah menunjukkan performa model yang lebih baik.  
3. **Root Mean Squared Error (RMSE)**  
- **Formula:** `RMSE = sqrt(MSE) = sqrt((1/n) * Σ(y_i - ŷ_i)²)`  
- **Penjelasan:** RMSE adalah akar kuadrat dari MSE. Keuntungannya adalah RMSE memiliki satuan yang sama dengan variabel target, sehingga lebih mudah diinterpretasikan dibandingkan MSE. Seperti MSE, RMSE juga memberikan bobot lebih pada error yang besar. Nilai RMSE yang lebih rendah menunjukkan performa model yang lebih baik dan merupakan salah satu metrik yang paling umum digunakan untuk mengevaluasi model regresi.  
4. **R-squared (R² atau Koefisien Determinasi)**  
- **Formula:** `R² = 1 - (SS_res / SS_tot)`  
1. `SS_res` (Sum of Squares of Residuals): `Σ(y_i - ŷ_i)²`  
2. `SS_tot` (Total Sum of Squares): `Σ(y_i - ȳ)²`, di mana `ȳ` adalah mean dari nilai aktual.  
- **Penjelasan:** R-squared mengukur proporsi varians dalam variabel dependen (target) yang dapat diprediksi dari variabel independen (fitur). Nilai R² berkisar antara 0 dan 1 (atau bisa negatif jika model sangat buruk). Nilai R² yang lebih tinggi (mendekati 1\) menunjukkan bahwa model lebih baik dalam menjelaskan variabilitas data. R² \= 0.75 berarti model dapat menjelaskan 75% varians pada data target.

### **Hasil Proyek Berdasarkan Metrik Evaluasi**

#### **Prediksi Interval 1 Menit**

Berikut adalah tabel hasil evaluasi dari ketiga model pada data uji untuk prediksi 1 menit:

| Model | MAE | MSE | RMSE | R-squared (R²) |
| ----- | ----- | ----- | ----- | ----- |
| SVM Regressor | 0.050948 | 0.004749 | 0.068915 | 0.884463 |
| Random Forest | 0.022472 | 0.001450 | 0.038077 | 0.964729 |
| XGBoost | 0.027126 | 0.002743 | 0.052376 | 0.933264 |

**Analisis Hasil (1 Menit):**

* Untuk prediksi dengan interval 1 menit, model **Random Forest** menunjukkan performa terbaik di semua metrik evaluasi. Model ini memiliki MAE terendah (0.022472), MSE terendah (0.001450), RMSE terendah (0.038077), dan R-squared tertinggi (0.964729).  
* Hasil ini mengindikasikan bahwa Random Forest paling akurat dalam memprediksi emisi CO2 untuk jangka pendek (1 menit) dan mampu menjelaskan sekitar 96.47% varians dalam data target.  
* SVM Regressor menunjukkan performa terburuk pada skenario ini, meskipun masih memiliki R-squared yang cukup baik (0.884463).  
* Visualisasi perbandingan nilai aktual vs. prediksi untuk setiap model juga dibuat untuk melihat seberapa baik model mengikuti pola data aktual.

#### **Prediksi Interval 1 Jam**

Berikut adalah tabel hasil evaluasi dari ketiga model pada data uji untuk prediksi 1 jam:

| Model | MAE | MSE | RMSE | R-squared (R²) |
| ----- | ----- | ----- | ----- | ----- |
| SVM Regressor | 0.079888 | 0.011128 | 0.105489 | 0.729485 |
| Random Forest | 0.089402 | 0.013316 | 0.115394 | 0.676302 |
| XGBoost | 0.095086 | 0.014940 | 0.122231 | 0.636805 |

**Analisis Hasil (1 Jam)**

* Untuk prediksi dengan interval 1 jam, model **SVM Regressor** menunjukkan performa terbaik di semua metrik evaluasi. Model ini memiliki MAE terendah (0.079888), MSE terendah (0.011128), RMSE terendah (0.105489), dan R-squared tertinggi (0.729485).  
* Ini menunjukkan bahwa SVM Regressor lebih unggul dalam memprediksi emisi CO2 untuk jangka waktu yang lebih panjang (1 jam) dibandingkan dua model lainnya, dengan kemampuan menjelaskan sekitar 72.95% varians data.  
* Pada skenario ini, XGBoost menunjukkan performa terendah. Perlu dicatat bahwa performa semua model menurun (R-squared lebih rendah) untuk prediksi 1 jam dibandingkan prediksi 1 menit, yang mengindikasikan tantangan yang lebih besar dalam memprediksi untuk horizon waktu yang lebih jauh.  
* Visualisasi perbandingan nilai aktual vs. prediksi juga dibuat untuk skenario ini.

#### **Ringkasan Pemilihan Model Berdasarkan Skor Komposit**
Berdasarkan analisis skor komposit yang memperhitungkan semua metrik:

* Model **Random Forest** (Skor: 12.0) dipilih sebagai model final untuk prediksi interval 1 menit.  
* Model **SVM Regressor** (Skor: 12.0) dipilih sebagai model final untuk prediksi interval 1 jam.

Temuan ini menarik karena model terbaik berbeda tergantung pada horizon waktu prediksi. Ini menunjukkan bahwa karakteristik data dan kemampuan model dalam menangkap pola jangka pendek dan jangka panjang dapat bervariasi.
