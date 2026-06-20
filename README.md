# tinyGPT-AllAboutRunning

Implementasi TinyGPT berbasis Transformer yang dilatih menggunakan corpus khusus bertema olahraga lari (running). Proyek ini bertujuan untuk membandingkan beberapa metode tokenisasi dan menganalisis pengaruhnya terhadap performa model bahasa dalam menghasilkan teks.

---

## Deskripsi Proyek

Proyek ini dibuat sebagai tugas mata kuliah Natural Language Processing (NLP).

Model TinyGPT dilatih menggunakan corpus bertema olahraga lari yang berisi lebih dari 2000 kata. Selanjutnya dilakukan perbandingan beberapa metode tokenisasi untuk mengetahui pengaruhnya terhadap proses pelatihan dan kualitas teks yang dihasilkan.

Metode tokenisasi yang digunakan:

* Character Tokenization
* Word Tokenization
* Byte Pair Encoding (BPE)

---

## Tujuan

* Membangun corpus bertema olahraga lari.
* Melatih model TinyGPT menggunakan corpus tersebut.
* Membandingkan beberapa metode tokenisasi.
* Menganalisis performa model berdasarkan hasil training.
* Menghasilkan teks menggunakan model yang telah dilatih.

---

## Struktur Proyek

```text
tinyGPT-AllAboutRunning/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── corpus.txt
│
├── models/
│   ├── char_model.pth
│   ├── word_model.pth
│   └── bpe_model.pth
│
├── tokenizer/
│   ├── tokenizer.model
│   └── tokenizer.vocab
│
└── src/
    ├── tinygpt.py
    └── transformer_blocks.py
```

---

## Arsitektur Model

Model menggunakan arsitektur TinyGPT yang terdiri dari:

* Token Embedding
* Positional Embedding
* Multi-Head Self Attention
* Feed Forward Network
* Layer Normalization
* Language Modeling Head

Model digunakan untuk melakukan prediksi token berikutnya (next token prediction).

---

## Instalasi

Clone repository:

```bash
git clone https://github.com/username/tinyGPT-AllAboutRunning.git
cd tinyGPT-AllAboutRunning
```

Install dependency:

```bash
pip install -r requirements.txt
```

---

## Cara Menjalankan Program

Masuk ke folder source code:

```bash
cd src
```

### 1. Training Model

#### Character Tokenization

```bash
python tinygpt.py train char
```

#### Word Tokenization

```bash
python tinygpt.py train word
```

#### BPE Tokenization

```bash
python tinygpt.py train bpe
```

Model hasil training akan disimpan pada folder:

```text
models/
```

---

### 2. Generate Teks

#### Generate menggunakan Character Tokenization

```bash
python tinygpt.py generate char
```

#### Generate menggunakan Word Tokenization

```bash
python tinygpt.py generate word
```

#### Generate menggunakan BPE Tokenization

```bash
python tinygpt.py generate bpe
```

Masukkan prompt ketika diminta:

```text
Masukkan prompt:
lari merupakan
```

Contoh hasil keluaran:

```text
lari merupakan salah satu cabang olahraga yang paling tua
dan paling populer di dunia. aktivitas ini telah dilakukan
manusia sejak zaman prasejarah sebagai bagian dari kebutuhan
untuk berburu, berpindah tempat, dan bertahan hidup.
```

---

## Author

Rifky Danu Asmoro

