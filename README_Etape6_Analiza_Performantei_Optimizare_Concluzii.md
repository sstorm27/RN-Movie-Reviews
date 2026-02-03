# README – Etapa 6: Analiza Performanței, Optimizarea și Concluzii Finale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** [Nume Prenume]  
**Link Repository GitHub: https://github.com/sstorm27/Proiect-RN 
**Data predării:** [04/02/2026]

---
## Scopul Etapei 6

Această etapă corespunde punctelor **7. Analiza performanței și optimizarea parametrilor**, **8. Analiza și agregarea rezultatelor** și **9. Formularea concluziilor finale**.

**Obiectiv principal:** Maturizarea completă a Sistemului cu Inteligență Artificială (SIA) prin optimizarea modelului RN (Bi-LSTM + Attention), analiza detaliată a performanței (focus pe Sarcasm) și integrarea îmbunătățirilor în aplicația software completă (Streamlit Dark Mode).

---

## 1. Actualizarea Aplicației Software în Etapa 6 

**CERINȚĂ CENTRALĂ:** Documentarea modificărilor aduse aplicației software ca urmare a optimizării modelului.

### Tabel Modificări Aplicație Software

| **Componenta** | **Stare Etapa 5** | **Modificare Etapa 6** | **Justificare** |
|----------------|-------------------|------------------------|-----------------|
| **Model încărcat** | `trained_model.h5` | `optimized_model.h5` | Integrare SpatialDropout și vocabular extins (20k). |
| **Interfață UI** | Standard Light | **Custom Dark Mode** | Aspect profesional și lizibilitate crescută. |
| **Feedback Vizual** | Text simplu | **Card-uri + Progress Bar** | Interpretare rapidă a scorului de încredere. |
| **Logic Injection** | Parțială | **Completă (64.79% date)** | Rezolvarea problemei de detectare a sarcasmului. |
| **Safety Net** | Inexistent | **Confidence Check** | Clasificarea scorurilor 0.45-0.55 ca "Neutru". |
| **Vocabular** | 15.000 cuvinte | **20.000 cuvinte** | Acoperire mai bună a argoului din datele reale. |

### Modificări concrete aduse în Etapa 6:

1. **Model înlocuit:** `models/trained_model.h5` → `models/optimized_model.h5`
   - Îmbunătățire: Accuracy +4.12%, F1 +0.06
   - Motivație: Modelul optimizat include `SpatialDropout1D` care previne memorarea cuvintelor cheie (ex: "best") în contexte sarcastice.

2. **State Machine actualizat:**
   - Stare nouă adăugată: `CONFIDENCE_CHECK` - Dacă scorul este între 0.45 și 0.55, sistemul afișează un avertisment "NEUTRU / MIXT" în loc să forțeze o decizie Pozitiv/Negativ.

3. **UI îmbunătățit:**
   - Implementare design **Dark Mode** cu accente Cyan/Green.
   - Adăugare **Sidebar** cu informații despre student și tehnologii.
   - Screenshot: `docs/screenshots/inference_optimized.png`

---

## 2. Analiza Detaliată a Performanței

### 2.1 Confusion Matrix și Interpretare

**Locație:** `docs/confusion_matrix_optimized.png`

### Interpretare Confusion Matrix:

**Clasa cu cea mai bună performanță:** **NEGATIV (0)**
- **Precision:** ~88%
- **Recall:** ~89%
- **Explicație:** Datorită injectării masive de date sintetice de tip "Sarcasm" și "Deception", modelul identifică excelent recenziile negative mascate sub formă de laude aparente.

**Clasa cu cea mai slabă performanță:** **NEUTRU (0.5)**
- **Precision:** ~82%
- **Recall:** ~79%
- **Explicație:** Zona de mijloc este subiectivă. Multe recenzii din dataset-ul real Kaggle etichetate ca "Pozitiv" sunt de fapt mediocre, ceea ce creează o ușoară confuzie cu clasa noastră artificială "Neutru".

**Confuzii principale:**
1. Clasa **Neutru** confundată cu clasa **Pozitiv** în ~12% din cazuri
   - **Cauză:** Subiectivitatea umană. O recenzie de tip "It was okay" poate fi considerată pozitivă de unii, neutră de alții.
   - **Impact industrial:** Acceptabil. Este mai grav să confunzi Negativ cu Pozitiv (ceea ce modelul evită cu succes).

### 2.2 Analiza Detaliată a 5 Exemple Greșite (Analiza Erorilor)

| **Index** | **Input (Rezumat)** | **Predicție** | **Real** | **Cauză probabilă** | **Soluție propusă** |
|-----------|---------------------|---------------|----------|---------------------|---------------------|
| #1 | "Film cu un buget redus dar inima mare" | NEUTRU | POZITIV | Cuvântul "redus" (low) are greutate negativă mare. | Augmentare cu fraze "low budget but good". |
| #2 | "Nu este genul meu, dar înțeleg de ce place." | NEGATIV | NEUTRU | "Nu este genul meu" domină semantic fraza. | Rafinare etichete pentru subiectivitate. |
| #3 | Recenzie plină de argou nou (slang gen Z). | NEUTRU | POZITIV | Cuvintele sunt OOV (Out of Vocabulary). | Creșterea vocabularului la 30k. |
| #4 | "Un film de nota 10... cu minus." | POZITIV | NEGATIV | Sarcasm foarte subtil, dependent de context cultural. | Mai multe date de tip "ironie fină". |
| #5 | Recenzie foarte scurtă ("Mda.") | NEUTRU | NEGATIV | Lipsă de context suficient pentru LSTM. | Ignorarea recenziilor < 3 cuvinte. |

---

## 3. Optimizarea Parametrilor și Experimentare

### 3.1 Strategia de Optimizare

**Abordare:** **Data-Centric AI** (Focus pe date, nu doar pe model).

**Axe de optimizare explorate:**
1. **Calitatea Datelor (Logic Injection):** Generarea a 46.000+ exemple sintetice pentru a acoperi cazurile limită (sarcasm, concesii).
2. **Arhitectură:** Trecerea de la LSTM simplu la **Bi-LSTM + Attention**.
3. **Regularizare:** Introducerea **SpatialDropout1D(0.3)** pentru a forța modelul să nu memoreze cuvinte specifice.
4. **Vocabular:** Extinderea de la 15k la 20k cuvinte.

### 3.2 Grafice Comparative
Fișiere generate în `docs/`: `loss_curve.png`, `confusion_matrix_optimized.png`.

### 3.3 Raport Final Optimizare (Tabel Experimente)

| **Exp#** | **Modificare față de Baseline** | **Accuracy** | **F1-score** | **Observații** |
|----------|--------------------------------|--------------|--------------|----------------|
| Baseline | LSTM Simplu (Etapa 3) | 0.72 | 0.68 | Nu detecta sarcasmul deloc. |
| Exp 2 | Bi-LSTM (Context bidirectional) | 0.78 | 0.74 | Mai bun, dar confuz la fraze lungi. |
| Exp 3 | Bi-LSTM + Attention | 0.82 | 0.79 | Overfitting rapid pe datele de train. |
| Exp 4 | + Logic Injection (Sarcasm Data) | 0.84 | 0.82 | Salt major în performanța pe sarcasm. |
| **FINAL** | **+ SpatialDropout + 12 Epoci** | **0.8642** | **0.8548** | **Modelul Final Optimizat.** |

**Justificare alegere configurație finală:**
Am ales **Exp 5 (FINAL)** deoarece combină capacitatea de înțelegere a contextului (Attention) cu robustețea la date noi (SpatialDropout). F1-Score-ul de **0.85** este singurul care satisface cerința industrială de a nu frustra utilizatorii cu clasificări greșite ale ironiei.

---

## 4. Agregarea Rezultatelor și Vizualizări

### 4.1 Tabel Sumar Rezultate Finale

| **Metrică** | **Etapa 5 (Baseline)** | **Etapa 6 (Final)** | **Target Industrial** | **Status** |
|-------------|------------------------|---------------------|-----------------------|------------|
| Accuracy | 82.30% | **86.42%** | ≥85% | ATINS |
| F1-score (macro) | 0.79 | **0.8548** | ≥0.80 | ATINS |
| Contribuție Date | 40% | **64.79%** | ≥40% | DEPĂȘIT |
| Latență inferență | 60ms | **~45ms** | ≤50ms | ATINS |

### 4.2 Vizualizări Obligatorii

Fișiere prezente în repository:
- `docs/confusion_matrix_optimized.png`
- `docs/loss_curve.png`
- `docs/screenshots/inference_optimized.png` (Screenshot UI Final)

---

## 5. Concluzii Finale și Lecții Învățate

### 5.1 Evaluarea Performanței Finale

**Obiective atinse:**
- [x] Construirea unui model robust la sarcasm (F1 > 0.85).
- [x] Contribuție originală majoră la date (64.79% date generate prin Logic Injection).
- [x] Aplicație Web profesională (Dark Mode, Feedback vizual).
- [x] Pipeline complet funcțional (End-to-End).

### 5.2 Limitări Identificate

1. **Dependența de Limbă:** Sistemul funcționează doar pentru limba engleză.
2. **Vocabular Fix:** Cuvintele care nu sunt în cele 20.000 cele mai frecvente sunt marcate ca `<OOV>` (Out of Vocabulary), pierzând informație.
3. **Subiectivitate:** Zona neutră (0.5) este interpretabilă și poate varia în funcție de utilizator.

### 5.3 Direcții de Cercetare și Dezvoltare

**Pe termen scurt:**
1. Implementarea unui mecanism de **Active Learning**: utilizatorul să poată corecta predicția în UI, iar modelul să învețe din asta.

**Pe termen mediu:**
1. Trecerea la o arhitectură **Transformer (BERT/RoBERTa)** pentru a capta nuanțe și mai fine, deși cu un cost computațional mai mare.

### 5.4 Lecții Învățate

1. **Datele sunt mai importante decât Modelul:** Am observat că îmbunătățirea setului de date (prin Logic Injection) a adus un câștig de performanță mai mare (+6%) decât schimbarea arhitecturii (+4%).
2. **Sarcasmul nu este magic:** Este un pattern statistic (Cuvânt Pozitiv + Context Negativ) care poate fi învățat dacă modelul vede suficiente exemple.
3. **Regularizarea este cheia:** Fără `SpatialDropout`, modelul memora doar cuvinte cheie ("best", "worst") și eșua la teste reale.

### 5.5 Plan Post-Feedback

Acest document reprezintă **VERSIUNEA FINALĂ PENTRU EXAMEN**.
Codul este înghețat ("Code Freeze"), iar modelul `optimized_model.h5` este cel care va fi prezentat în demonstrația live.

---

## Structura Repository-ului Final

proiect-rn-[nume-prenume]/ ├── README.md # Overview Final ├── etapa6_optimizare_concluzii.md # ← ACEST FIȘIER ├── docs/ │ ├── confusion_matrix_optimized.png # Matricea de confuzie │ ├── loss_curve.png # Grafic antrenare │ └── screenshots/ │ └── inference_optimized.png # Screenshot UI Final ├── data/generated/ # Datele sintetice ├── src/ │ ├── neural_network/ │ │ ├── train.py # Script antrenare Final │ │ ├── model.py # Arhitectură Bi-LSTM + SpatialDropout │ │ └── attention.py # Layer Custom │ └── app/ │ └── main.py # UI Streamlit Final ├── models/ │ └── optimized_model.h5 # MODEL FINAL ├── results/ │ ├── training_history.csv # Log antrenare │ └── final_metrics.json # Metrici JSON └── requirements.txt
