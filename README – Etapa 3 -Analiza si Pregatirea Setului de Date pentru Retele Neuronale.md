# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Ionescu David  
**Data:**

---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, concentrându-se pe pregătirea setului de date pentru sistemul de Analiză a Sentimentelor. Deoarece seturile de date standard (ex: IMDB) conțin erori de etichetare și nu acoperă nuanțe lingvistice complexe (sarcasm, opinii concesive), am optat pentru o strategie de **augmentare sintetică controlată ("Logic Injection")**, combinând date reale cu date generate programatic pentru a forța modelul să învețe tipare logice specifice.

---

##  1. Structura Repository-ului Github (versiunea Etapei 3)

project-rn/ ├── README.md ├── docs/ │ └── datasets/ # grafice distribuție clase ├── data/ │ ├── raw/ # IMDB Dataset.csv (date brute) │ ├── generated/ # Date generate sintetic (sarcasm, logică) │ ├── processed/ # Date tokenizate și curățate │ ├── train/ # Set de antrenare (85%) │ ├── validation/ # Set de validare (15%) │ └── test/ # Set de testare (inclus în validare pentru rapiditate) ├── src/ │ ├── preprocessing/ # Tokenizer, Padding sequences │ ├── data_acquisition/ # Scriptul de generare (train.py) │ └── neural_network/ # Modelul Bi-LSTM + Attention ├── config/ │ └── tokenizer.pkl # Dicționarul de cuvinte salvat └── requirements.txt # tensorflow, pandas, numpy, streamlit


---

##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** Hibridă.
    1. **Baza:** Dataset public IMDB (Kaggle) - recenzii de film reale.
    2. **Augmentare (Majoritară):** Generare programatică folosind scripturi Python proprii (`src/neural_network/train.py`).
* **Modul de achiziție:** ☑ Generare programatică (Logic Injection) + ☑ Fișier extern (Kaggle).
* **Motivație:** Datele reale nu conțineau suficiente exemple de sarcasm ("best cure for insomnia") sau structuri complexe ("boring start but great ending"), ducând la erori de context.

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** ~45.000 (variabil în funcție de parametrii de generare).
* **Număr de caracteristici (features):** 1 (Textul recenziei) -> transformat în secvență de 200 intregi.
* **Tipuri de date:** ☑ Text (NLP) / ☑ Numerice (Scoruri sentiment).
* **Format fișiere:** CSV (pentru stocare) și Pandas DataFrame (în memorie).

### 2.3 Descrierea etichetelor (Target)

| **Etichetă (Score)** | **Semnificație** | **Exemplu** |
|-------------------|------------------|-------------|
| **0.0** | Negativ | "This movie is a waste of time." / "Best cure for insomnia." |
| **0.5** | Neutru / Average | "It was an average movie, nothing special." |
| **1.0** | Pozitiv | "A masterpiece." / "Boring start but amazing ending." |

---

##  3. Analiza Exploratorie a Datelor (EDA)

### 3.1 Statistici descriptive

* **Lungimea medie a recenziilor:** Variabilă (de la 3 cuvinte la 500+ cuvinte).
* **Vocabular:** Am limitat vocabularul la cele mai frecvente **15.000 de cuvinte** pentru a elimina zgomotul (nume proprii rare, greșeli de tipar).
* **Distribuția claselor:**
    * Inițial (Kaggle): Puternic polarizat (doar Pozitiv/Negativ).
    * Final (Hibrid): Echilibrat artificial pentru a include clasa Neutră și cazurile de Sarcasm ("Edge cases").

### 3.2 Probleme identificate în datele brute (Raw Data)

* **Lipsa Zonei Neutre:** Dataset-ul IMDB forțează recenziile de nota 5 sau 6 în categoriile "Negativ" sau "Pozitiv", creând confuzie modelului.
* **Orbire la Context:** Cuvintele "Best", "Great", "Cure" apar frecvent în recenzii negative sarcastice, dar statistic sunt asociate cu clasa pozitivă.
* **Contaminare:** Expresii precum "Not bad" erau adesea etichetate greșit în dataset-urile automate.

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea și Generarea Datelor (Data Cleaning & Generation)

În loc să curățăm manual datele eronate, am aplicat o strategie de **Generare Controlată**:
* **Happy End Scenarios:** Am generat fraze de tip "Start Rău -> Final Bun" etichetate corect (1.0).
* **Sarcasm Injection:** Am generat mii de exemple de tip "Watch paint dry" etichetate corect (0.0).
* **Tratarea valorilor lipsă:** Nu există (datele sunt generate sau curățate la citire).

### 4.2 Transformarea caracteristicilor (NLP Pipeline)

* **Tokenizare:** Transformarea textului în numere folosind un `Tokenizer` antrenat pe corpus (max_words=15000). Caracterul `<OOV>` este folosit pentru cuvinte necunoscute.
* **Padding:** Uniformizarea secvențelor la lungimea fixă de **200 de tokeni** (padding='post', truncating='post') pentru a fi compatibile cu intrarea rețelei LSTM.
* **Encoding:** Etichetele sunt valori float continue (0.0 - 1.0) pentru a permite regresia sentimentului (inclusiv zona gri 0.5).

### 4.3 Structurarea seturilor de date

**Împărțire:**
* **Train:** ~85% (Prioritizăm volumul mare pentru a expune modelul la variații de sarcasm).
* **Validation/Test:** ~15% (Folosit pentru monitorizarea `val_loss` și Early Stopping).

**Principii respectate:**
* **Data Leakage:** Generarea datelor de test se face separat sau prin `train_test_split` cu seed fix (`random_state=42`) pentru reproductibilitate.

### 4.4 Salvarea rezultatelor

* Tokenizer-ul este salvat în `config/tokenizer.pkl` pentru a fi folosit identic în aplicația de inferență (UI).
* Modelul antrenat este salvat în `models/optimized_model.h5`.

---

##  5. Fișiere Generate în Această Etapă

* `src/neural_network/train.py` – Scriptul principal care combină generarea datelor cu preprocesarea și antrenarea.
* `config/tokenizer.pkl` – Obiectul de preprocesare salvat.
* `data/processed/kaggle_combined.csv` – Subsetul de date reale curățate.

---

##  6. Stare Etapă

- [x] Structură repository configurată
- [x] Dataset analizat (Identificat lipsa sarcasmului și a clasei neutre)
- [x] Date preprocesate (Tokenizare + Padding)
- [x] Date augmentate (Logic Injection pentru sarcasm)
- [x] Seturi train/val generate
- [x] Documentație actualizată

---
