# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Ionescu David  
**Data:** 21.01.2026
---

## Scopul Etapei 4

Această etapă corespunde punctului **5. Dezvoltarea arhitecturii aplicației software bazată pe RN**.
Am livrat un **SCHELET COMPLET și FUNCȚIONAL** al întregului Sistem cu Inteligență Artificială (SIA). Toate modulele sunt interconectate și funcționale.

---

##  Livrabile Obligatorii

### 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru** | **Modul software responsabil** |
|---------------------------|--------------------------------|--------------------------------|
| Interpretarea corectă a recenziilor sarcastice ("Best cure for insomnia") unde modelele clasice eșuează | Arhitectură **Bi-LSTM + Attention** care analizează contextul global al frazei, nu doar cuvinte cheie | **Modul 2: Neural Network** (`model.py`, `attention.py`) |
| Generarea unui dataset echilibrat care să conțină nuanțe ("Average", "Not bad") și structuri logice complexe | Algoritm de **"Logic Injection"** care generează sintetic mii de exemple de structuri concesive ("Even though...") | **Modul 1: Data Acquisition** (`train.py`) |
| Feedback vizual și interpretare instantanee a sentimentului pentru utilizatori non-tehnici | Interfață Web (Streamlit) cu bare de progres și coduri de culoare (Verde/Galben/Roșu) în funcție de scor | **Modul 3: Web Service / UI** (`main.py`) |

---

### 2. Contribuția Originală la Setul de Date – MINIM 40% din Totalul Observațiilor Finale

### Contribuția originală la setul de date:

**Total observații finale:** ~45.000 (după Etapa 3 + Etapa 4)
**Observații originale:** ~35.000 (~75-80%)

**Tipul contribuției:**
[X] Date sintetice prin metode avansate (Logic Injection & Pattern Generation)

**Descriere detaliată:**
Deoarece dataset-urile publice (IMDB) sunt binare și nu conțin suficiente exemple de sarcasm sau opinii moderate ("zona gri"), am dezvoltat un generator de date în Python. Acesta nu face doar o simplă augmentare, ci construiește fraze noi combinând șabloane gramaticale conflictuale (ex: "Început Rău" + "Conector Adversativ (DAR)" + "Final Bun" => Etichetă Pozitivă). Aceasta forțează modelul să învețe logica frazei, nu doar vocabularul.

**Locația codului:** `src/neural_network/train.py` (Funcția `generate_smart_data`)
**Locația datelor:** Generate dinamic și salvate în memorie sau `data/generated/` (dacă se activează exportul).

**Dovezi:**
- Scriptul `audit_project.py` (creat în etapa anterioară) demonstrează procentul de date generate vs. reale.

---

### 3. Diagrama State Machine a Întregului Sistem (OBLIGATORIE)

**Diagrama conceptuală a fluxului de date:**

IDLE → USER_INPUT → PREPROCESS (Tokenize & Pad) → RN_INFERENCE (Bi-LSTM) → ATTENTION_WEIGHTING → HEURISTIC_CHECK (Safety Net) → ├─ [Score > 0.55] → DISPLAY_POSITIVE (Green) ├─ [Score < 0.45] → DISPLAY_NEGATIVE (Red) └─ [Score 0.45-0.55] → DISPLAY_NEUTRAL (Yellow) ↓ LOG_RESULT → IDLE


**Justificarea State Machine-ului ales:**

Am ales o arhitectură de tip **Pipeline de Procesare Secvențială cu Safety Net** pentru a gestiona complexitatea limbajului natural.

Stările principale sunt:
1. **PREPROCESS:** Transformarea textului în secvențe numerice de lungime fixă (200), esențială pentru LSTM.
2. **RN_INFERENCE:** Rularea modelului neural principal.
3. **HEURISTIC_CHECK:** Aceasta este o stare critică adăugată pentru robustețe industrială. Deși modelul neural este puternic, anumite expresii idiomatice rare ("cure for insomnia") pot fi interpretate greșit. Această stare aplică reguli logice (RegEx) post-inferență pentru a corecta eventualele scăpări grave ale AI-ului înainte de afișare.

---

### 4. Scheletul Complet al celor 3 Module Cerute

Am implementat un schelet complet funcțional în Python:

| **Modul** | **Fișiere / Locație** | **Descriere Funcțională** |
|-----------|-----------------------|---------------------------|
| **1. Data Logging / Acquisition** | `src/neural_network/train.py` | Script care descarcă datele reale, le combină cu cele generate sintetic și pregătește vectorii pentru antrenament. Rulează fără erori. |
| **2. Neural Network Module** | `src/neural_network/model.py` <br> `src/neural_network/attention.py` | Definește arhitectura Bi-LSTM și stratul custom de Atenție. Modelul este compilat și gata de antrenare. Include suport pentru salvare/încărcare `.h5`. |
| **3. Web Service / UI** | `src/app/main.py` | Aplicație Streamlit care încarcă modelul și tokenizer-ul, preia input de la tastatură și afișează rezultatul clasificării în timp real. |

---

## Structura Repository-ului la Finalul Etapei 4

proiect-rn-ionescu-david/ ├── data/ │ ├── raw/ # Dataset IMDB original │ ├── generated/ # Datele sintetice (Logic Injection) │ ├── processed/ # Tokenizer cache │ └── train/ # Split-uri de date ├── src/ │ ├── data_acquisition/ # (Integrat în train.py pentru eficiență) │ ├── neural_network/ │ │ ├── train.py # Modul 1 (Generare & Antrenare) │ │ ├── model.py # Modul 2 (Definiție Arhitectură) │ │ └── attention.py # Modul 2 (Layer Custom) │ └── app/ │ └── main.py # Modul 3 (UI Streamlit) ├── docs/ │ ├── state_machine.png # Diagrama fluxului │ └── screenshots/ # Demonstrație UI ├── models/ │ └── untrained_model.h5 # Model inițializat ├── config/ │ └── tokenizer.pkl ├── README.md ├── README_Etapa3.md ├── README_Etapa4_Arhitectura_SIA.md # ← ACEST FIȘIER └── requirements.txt


---

## Checklist Final

### Documentație și Structură
- [x] Tabelul Nevoie → Soluție → Modul complet
- [x] Declarație contribuție >40% date originale completată (Date sintetice avansate)
- [x] Diagrama State Machine definită și justificată
- [x] Repository structurat corect

### Module Funcționale
- [x] **Modul 1:** `train.py` rulează și generează datele hibride.
- [x] **Modul 2:** `model.py` definește corect arhitectura Bi-LSTM + Attention.
- [x] **Modul 3:** `main.py` pornește interfața web și acceptă input.

---
