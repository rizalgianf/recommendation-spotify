# Laporan Proyek Machine Learning - Sistem Rekomendasi Musik Spotify - Rizal Gian Febriantama

## Project Overview

Musik adalah bagian penting dalam kehidupan sehari-hari. Platform streaming seperti Spotify menyediakan jutaan lagu, namun banyaknya pilihan membuat pengguna kesulitan menemukan lagu yang sesuai preferensi. Sistem rekomendasi musik sangat dibutuhkan untuk membantu pengguna menemukan lagu yang relevan dan menarik.

Sistem rekomendasi berbasis machine learning dapat meningkatkan pengalaman pengguna dengan memberikan saran lagu yang dipersonalisasi. Dengan memanfaatkan fitur audio, genre, dan metadata artis, sistem dapat mengidentifikasi kemiripan antar lagu dan memberikan rekomendasi yang lebih akurat.

**Referensi:**
- Harjananto, D. Y., Kartika Dewi, R., & Brata, K. C. (2021). Pengembangan Sistem Rekomendasi Musik berdasarkan Waktu berbasis Android. Jurnal Pengembangan Teknologi Informasi Dan Ilmu Komputer, 5(5), 1729–1733. http://j-ptiik.ub.ac.id
- Muhyidin, M. S., Hariyanti, I., Novianto, M. F., & Alifia, A. (n.d.). ANALISIS SISTEM REKOMENDASI MUSIK BERDASARKAN LIRIK DENGAN METODE TERM FREQUENCY-INVERSE DOCUMENT FREQUENCY ( TF – IDF ).

---

## Business Understanding

### Problem Statements

- Bagaimana membangun sistem rekomendasi lagu yang dapat memberikan saran lagu serupa berdasarkan lagu yang dipilih pengguna?

### Goals

- Menghasilkan sistem rekomendasi lagu berbasis content-based filtering yang mampu memberikan rekomendasi lagu serupa.

### Solution Approach

#### Solution Statements

1. **Content-Based Filtering**  
   Menggunakan kemiripan fitur konten (audio, genre, artis) untuk merekomendasikan lagu yang mirip dengan lagu input.

---

## Data Understanding

Dataset yang digunakan adalah [Top 10,000 Spotify Songs (1960-now)](https://www.kaggle.com/datasets/joebeachcapital/top-10000-spotify-songs-1960-now) dari Kaggle, berisi 10.000 lagu populer dari tahun 1950 hingga sekarang.

**Jumlah Data:**  
- Baris: 10.000  
- Kolom: 35

**Fitur utama:**
- Track Name: Nama lagu
- Artist Name(s): Nama artis
- Artist Genres: Genre artis
- Danceability, Energy, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo: Fitur audio numerik
- Popularity: Skor popularitas lagu

**Kondisi Data:**
    - Sebagian besar kolom memiliki data yang cukup lengkap, namun terdapat beberapa kolom dengan nilai kosong (missing values):
      - **Kolom `Album Genres`** sepenuhnya kosong (10.000 nilai kosong).
      - **Kolom `Artist Genres`** memiliki 551 nilai kosong.
      - **Kolom `Track Preview URL`** memiliki 63 nilai kosong.
      - Kolom numerik seperti `Danceability`, `Energy`, `Key`, `Loudness`, `Mode`, `Speechiness`, `Acousticness`, `Instrumentalness`, `Liveness`, `Valence`, `Tempo`, dan `Time Signature` memiliki masing-masing 5 nilai kosong.
      - Kolom metadata seperti `Label` dan `Copyrights` memiliki masing-masing 7 dan 23 nilai kosong.

**Insight Data:**
- Lagu berasal dari berbagai dekade dan genre.
- Fitur audio lengkap, cocok untuk analisis karakteristik musik dan sistem rekomendasi.

---

## Data Preparation

### 1. Pemeriksaan dan Pembersihan Data

```python
print("Jumlah baris:", df.shape[0])
print("Jumlah kolom:", df.shape[1])
df.info()
print(df.isnull().sum())
```
**Penjelasan:**  
Kode di atas digunakan untuk mengetahui jumlah baris, kolom, tipe data, dan jumlah missing value pada setiap kolom. Ini penting untuk menentukan langkah pembersihan data selanjutnya.

### 2. Menghapus Missing Value pada Fitur Penting

```python
df_clean = df.dropna(subset=[
    'Track Name', 'Artist Name(s)', 'Artist Genres',
    'Danceability', 'Energy', 'Speechiness', 'Acousticness',
    'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Popularity'
]).reset_index(drop=True)
```
**Penjelasan:**  
Baris dengan nilai kosong pada fitur penting dihapus agar data yang digunakan lengkap dan valid.

### 3. Gabungkan Fitur Teks (Genre dan Artis)

```python
df_clean['combined_features'] = (
    df_clean['Artist Genres'].astype(str) + ' ' +
    df_clean['Artist Name(s)'].astype(str)
)
```
**Penjelasan:**  
### Penjelasan Kode: Gabungkan Fitur Teks (Genre dan Artis)

Kode ini bertujuan untuk membuat kolom baru bernama `combined_features` dalam DataFrame `df_clean`. Kolom ini menggabungkan informasi dari dua kolom teks, yaitu `Artist Genres` dan `Artist Name(s)`. Berikut penjelasan langkah-langkahnya:

 **`df_clean['Artist Genres'].astype(str)`**  
    Mengonversi nilai dalam kolom `Artist Genres` menjadi tipe data string. Hal ini dilakukan untuk memastikan semua nilai dapat digabungkan, termasuk nilai NaN (akan diubah menjadi string `'nan'`).

 **`df_clean['Artist Name(s)'].astype(str)`**  
    Mengonversi nilai dalam kolom `Artist Name(s)` menjadi tipe data string, dengan alasan yang sama seperti langkah sebelumnya.

 **Penggabungan dengan Operator `+`**  
    Menggabungkan string dari kolom `Artist Genres` dan `Artist Name(s)` dengan menambahkan spasi (`' '`) di antara keduanya. Hasilnya adalah string gabungan yang berisi genre dan nama artis.

 **Hasil Akhir**  
    Kolom baru `combined_features` akan berisi teks gabungan dari genre dan nama artis, yang dapat digunakan untuk analisis berbasis teks, seperti pembuatan matriks TF-IDF untuk sistem rekomendasi.

**Contoh Hasil:**
- Jika `Artist Genres` = `"pop, uk pop"` dan `Artist Name(s)` = `"Ed Sheeran"`, maka `combined_features` = `"pop, uk pop Ed Sheeran"`.
- Jika `Artist Genres` = `NaN` dan `Artist Name(s)` = `"Adele"`, maka `combined_features` = `"nan Adele"`.


### 4. Pilih dan Normalisasi Fitur Numerik

```python
num_features = [
    'Danceability','Energy','Speechiness','Acousticness',
    'Instrumentalness','Liveness','Valence','Tempo','Popularity'
]
scaler = MinMaxScaler()
df_num_scaled = scaler.fit_transform(df_clean[num_features])
```
**Penjelasan:**  
### Penjelasan Kode: Pilih dan Normalisasi Fitur Numerik

**`num_features`**  
    - Variabel `num_features` adalah daftar nama kolom yang berisi fitur numerik dari DataFrame `df_clean`. Fitur-fitur ini dipilih karena relevan untuk analisis atau pemrosesan lebih lanjut.
    - Fitur yang dipilih:
      - `Danceability`: Tingkat kelayakan lagu untuk menari.
      - `Energy`: Tingkat energi lagu.
      - `Speechiness`: Proporsi elemen vokal dalam lagu.
      - `Acousticness`: Kemungkinan lagu bersifat akustik.
      - `Instrumentalness`: Tingkat instrumental dalam lagu.
      - `Liveness`: Kemungkinan lagu direkam secara langsung.
      - `Valence`: Tingkat kebahagiaan atau kesedihan lagu.
      - `Tempo`: Kecepatan lagu dalam BPM (beats per minute).
      - `Popularity`: Popularitas lagu berdasarkan Spotify.

 **`scaler = MinMaxScaler()`**  
    - Membuat objek `MinMaxScaler` dari library `sklearn.preprocessing`.
    - `MinMaxScaler` digunakan untuk normalisasi data, yaitu mengubah nilai fitur ke dalam rentang [0, 1]. Hal ini dilakukan agar semua fitur memiliki skala yang sama, sehingga tidak ada fitur yang mendominasi analisis atau model.

 **`df_clean[num_features]`**  
    - Mengambil subset DataFrame `df_clean` yang hanya berisi kolom-kolom yang ada di `num_features`.

 **`scaler.fit_transform(df_clean[num_features])`**  
    - `fit_transform()` melakukan dua hal:
      - **`fit`**: Menghitung nilai minimum dan maksimum dari setiap kolom dalam `num_features`.
      - **`transform`**: Mengubah nilai setiap kolom ke dalam rentang [0, 1] berdasarkan nilai minimum dan maksimum yang telah dihitung.
    - Hasilnya adalah array numpy (`df_num_scaled`) yang berisi nilai-nilai fitur numerik yang telah dinormalisasi.

 **`df_num_scaled`**  
    - Variabel ini menyimpan hasil normalisasi dalam bentuk array numpy. Array ini dapat digunakan untuk analisis lebih lanjut, seperti pembuatan matriks fitur gabungan atau input ke model machine learning.

### Contoh:
Jika nilai awal kolom `Danceability` adalah [0.5, 0.7, 0.9], maka setelah normalisasi dengan `MinMaxScaler`, nilai tersebut akan diubah menjadi [0.0, 0.5, 1.0] (dengan asumsi 0.5 adalah nilai minimum dan 0.9 adalah nilai maksimum).


### 5. Gabungkan Fitur Teks dan Numerik

```python

# Create TF-IDF vectorizer for text features
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_clean['combined_features'])

combined_matrix = hstack([tfidf_matrix, df_num_scaled])
combined_matrix = csr_matrix(combined_matrix)

```
**Penjelasan:**  
### Penjelasan Kode: Gabungkan Fitur Teks dan Numerik

1. **TF-IDF Vectorization pada Fitur Teks**
  - `tfidf = TfidfVectorizer(stop_words='english')`  
    Membuat objek TF-IDF vectorizer untuk mengubah data teks menjadi representasi numerik, dengan menghilangkan stopwords bahasa Inggris.
  - `tfidf_matrix = tfidf.fit_transform(df_clean['combined_features'])`  
    Mengubah kolom `combined_features` (gabungan genre dan nama artis) menjadi matriks TF-IDF. Setiap lagu direpresentasikan sebagai vektor berdasarkan kata-kata unik yang muncul di seluruh dataset.

2. **Penggabungan Fitur Teks dan Numerik**
  - `combined_matrix = hstack([tfidf_matrix, df_num_scaled])`  
    Menggabungkan matriks TF-IDF (fitur teks) dengan array hasil normalisasi fitur numerik (`df_num_scaled`) secara horizontal, sehingga setiap lagu memiliki representasi fitur gabungan (teks + numerik).
  - `combined_matrix = csr_matrix(combined_matrix)`  
    Mengubah hasil gabungan menjadi format sparse matrix (Compressed Sparse Row) agar efisien dalam penyimpanan dan komputasi, terutama untuk data berdimensi besar.

**Kesimpulan:**  
Kode ini menghasilkan matriks fitur gabungan yang siap digunakan untuk menghitung kemiripan antar lagu pada sistem rekomendasi berbasis content-based filtering.
=======
 **`hstack([tfidf_matrix, df_num_scaled])`**  
    - Fungsi `hstack` dari `scipy.sparse` digunakan untuk menggabungkan dua matriks secara horizontal (kolom demi kolom).
    - **`tfidf_matrix`**: Matriks TF-IDF yang merepresentasikan fitur teks (`combined_features`) dalam bentuk vektor sparse. Matriks ini memiliki dimensi `(9446, 4483)`.
    - **`df_num_scaled`**: Matriks numpy yang berisi fitur numerik yang telah dinormalisasi. Matriks ini memiliki dimensi `(9446, 9)`.
    - Hasil penggabungan adalah matriks sparse dengan dimensi `(9446, 4492)`.

 **`csr_matrix(combined_matrix)`**  
    - Fungsi `csr_matrix` dari `scipy.sparse` digunakan untuk mengonversi hasil penggabungan menjadi format **Compressed Sparse Row (CSR)**.
    - Format CSR lebih efisien untuk penyimpanan dan operasi matematis pada matriks sparse, seperti perhitungan kemiripan menggunakan `cosine_similarity`.

 **Hasil Akhir**  
    - Variabel `combined_matrix` adalah matriks sparse dengan dimensi `(9446, 4492)`, yang menggabungkan fitur teks dan numerik.
    - Matriks ini dapat digunakan sebagai input untuk analisis lebih lanjut, seperti perhitungan kemiripan antar lagu atau model machine learning.

### Contoh Dimensi:
- Sebelum penggabungan:
  - `tfidf_matrix`: `(9446, 4483)`
  - `df_num_scaled`: `(9446, 9)`
- Setelah penggabungan:
  - `combined_matrix`: `(9446, 4492)`

---
>>>>>>> 82cd89a38e5c435c86f550a57d03b94093e6a977

## Modeling

### Content-Based Filtering

```python
def recommend(track_name, top_n=5):
    idx = df_clean[df_clean['Track Name'].str.lower() == track_name.lower()].index
    if len(idx) == 0:
        return "Track tidak ditemukan."
    idx = idx[0]
    sim_scores = cosine_similarity(combined_matrix[idx], combined_matrix).flatten()
    sim_indices = sim_scores.argsort()[-top_n-1:-1][::-1]
    return df_clean.iloc[sim_indices][['Track Name', 'Artist Name(s)', 'Artist Genres']]
```
**Penjelasan:**  
## Sistem Rekomendasi Musik Berbasis Content-Based Filtering

Sistem rekomendasi ini bekerja dengan menggunakan teknik Content-Based Filtering untuk merekomendasikan lagu-lagu serupa berdasarkan karakteristik konten lagu. Berikut penjelasan cara kerjanya:

 **Pencarian Lagu**  
    Fungsi `recommend()` mencari lagu dalam dataset berdasarkan nama lagu yang dimasukkan pengguna.

 **Perhitungan Kesamaan**  
    Setelah lagu ditemukan, sistem menghitung skor kesamaan antara lagu tersebut dengan semua lagu lain dalam dataset menggunakan cosine similarity.

 **Pemilihan Rekomendasi**  
    Sistem memilih lagu-lagu dengan skor kesamaan tertinggi sebagai rekomendasi, dengan jumlah sesuai parameter `top_n`.

 **Hasil Rekomendasi**  
    Output berupa dataframe yang menampilkan informasi lagu-lagu yang direkomendasikan, meliputi judul lagu, nama artis, dan genre.

Fitur yang digunakan mencakup kombinasi fitur teks (genre dan nama artis) dan fitur audio (seperti danceability, energy, speechiness, dll) yang telah dinormalisasi untuk menghasilkan rekomendasi yang akurat berdasarkan karakteristik musik.

**Contoh Penggunaan:**
```python
hasil = recommend('Shape of You')
print(hasil)
```
## Insight Hasil Rekomendasi untuk Lagu "Shape of You"

### Analisis Rekomendasi
 **Kesamaan Artis**: Sistem merekomendasikan 5 lagu yang seluruhnya dari artis yang sama, yaitu Ed Sheeran. Ini menunjukkan bahwa model mengidentifikasi kesamaan creator sebagai faktor signifikan dalam preferensi musik.

 **Konsistensi Genre**: Semua lagu yang direkomendasikan memiliki genre yang identik: "pop, singer-songwriter pop, uk pop". Hal ini menandakan sistem berhasil mengidentifikasi karakteristik genre yang relevan dengan lagu input.

 **Variasi Lagu**: Sistem merekomendasikan berbagai lagu dari Ed Sheeran dengan judul berbeda ("Shivers", "Eyes Closed", "New York", "Small Bump", "Sing"), menunjukkan keragaman dalam katalog musik artis tersebut.

 **Efektivitas Content-Based Filtering**: Hasil ini memperlihatkan bahwa sistem rekomendasi berbasis konten bekerja dengan baik dalam mengidentifikasi kesamaan berdasarkan metadata dan fitur audio.

 **Potensi Pengembangan**: Meskipun rekomendasi akurat dari segi kesamaan artis dan genre, sistem mungkin dapat ditingkatkan untuk memberikan variasi artis yang lebih beragam namun tetap mempertahankan kesamaan karakteristik musik.

**Kelebihan:**
- Tidak membutuhkan data interaksi pengguna.
- Dapat memberikan rekomendasi lagu baru yang belum pernah didengar pengguna.

**Kekurangan:**
- Rekomendasi cenderung terbatas pada lagu-lagu dengan fitur serupa (misal: artis atau genre yang sama).
- Tidak mempertimbangkan selera kolektif pengguna lain.

---

## Evaluation

Metrik evaluasi utama adalah **cosine similarity** antara fitur lagu input dan lagu lain. Evaluasi dilakukan secara kualitatif dengan melihat relevansi hasil rekomendasi.

## Analisis Hasil Evaluasi Sistem Rekomendasi Musik

Hasil evaluasi menunjukkan kinerja sistem rekomendasi musik berbasis content-based filtering untuk empat lagu populer. Berikut analisisnya:

### 1. Artist Diversity (Keberagaman Artis)
- **Nilai rata-rata: 0.25 (skala 0-1)**
- Nilai ini cukup rendah, menunjukkan sistem cenderung merekomendasikan lagu dari artis yang sama
- 3 dari 4 lagu ("Shape of You", "Billie Jean", "Watermelon Sugar") memiliki nilai 0.2, artinya hanya 20% artis yang unik
- "Bad Guy" memiliki keberagaman artis lebih tinggi (0.4), menandakan variasi artis yang lebih baik

### 2. Genre Diversity (Keberagaman Genre)
- **Nilai rata-rata: 0.21 (skala 0-1)**
- Nilai ini rendah, mengindikasikan rekomendasi terbatas pada genre yang sangat mirip
- "Bad Guy" menunjukkan diversitas genre tertinggi (0.25)
- Lagu lainnya memiliki nilai 0.20, menandakan kurangnya variasi genre

### 3. Average Popularity (Rata-rata Popularitas)
- **Nilai rata-rata: 50.90 (skala 0-100)**
- "Billie Jean" mendapatkan rekomendasi lagu-lagu paling populer (69.8)
- "Shape of You" justru mendapat rekomendasi lagu-lagu kurang populer (28.6)
- Menunjukkan sistem tidak selalu mengutamakan lagu populer, tetapi lebih fokus pada kesamaan karakteristik

### 4. Average Similarity (Rata-rata Kesamaan)
- **Nilai rata-rata: 0.94 (skala 0-1)**
- Nilai sangat tinggi, menunjukkan rekomendasi sangat relevan dari segi kemiripan konten
- "Billie Jean" memiliki tingkat kemiripan tertinggi (0.988), hampir sempurna
- Semua lagu mendapatkan rekomendasi dengan nilai kesamaan di atas 0.9, mengindikasikan sistem sangat baik dalam mengidentifikasi lagu-lagu yang mirip

### Kesimpulan
Sistem rekomendasi sangat kuat dalam memberikan rekomendasi yang relevan (similarity tinggi) tetapi kurang dalam keberagaman (diversity rendah). Ini merupakan trade-off klasik dalam sistem rekomendasi content-based, di mana relevansi tinggi sering berarti keberagaman rendah. Untuk peningkatan, dapat dipertimbangkan teknik hybrid yang menggabungkan content-based dengan collaborative filtering atau menambahkan faktor randomisasi tertentu untuk meningkatkan keberagaman rekomendasi.

