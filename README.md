# Projekt 1 – scoring/rating PD: interpretowalność i wyjaśnialność modeli

Projekt zaliczeniowy z przedmiotu *Interpretowalność i Wyjaśnialność Uczenia Maszynowego*.
**Autorzy:** Filip Kowalczyk, Paweł Stefańczyk

Model oceny ryzyka niewypłacalności (PD) firm na podstawie danych finansowych. Porównujemy
interpretowalną regresję logistyczną z modelem black-box (XGBoost) i analizujemy ich
wyjaśnialność (SHAP, LIME, PDP/ICE/ALE, surogat globalny, kontrfaktyk), kalibrację do
zadanego poziomu PD (4%), skalę ratingową, progi decyzyjne i monitoring.

Szczegółowy opis modeli, wyników i ograniczeń: [**MODEL\_CARD.md**](MODEL_CARD.md).

## Najważniejsze wyniki (zbiór testowy, n = 600, 81 defaultów)

||LR (L1, C = 0,03)|XGBoost|
|-|-|-|
|AUC|0,720|**0,771**|
|Gini|0,440|0,541|
|KS|0,354|0,422|

Różnica Gini XGB − LR = +0,101, CI95 \[+0,021; +0,176]. Główne wnioski: TreeSHAP jest ściśle
addytywny i stabilny, a LIME ma umiarkowaną wierność (R² poza próbą ≈ 0,58). Kalibracja do 4%
zaniża PD ok. 3,3× względem obserwowanego DR (13,5%), co jest założeniem biznesowym, a nie
poprawą kalibracji statystycznej (szczegóły w model card).

## Struktura repozytorium

```
.
├── final.ipynb   # cały projekt: dane → modele → wyjaśnienia → ocena końcowa
├── zbiór_3.csv                 # dane
├── MODEL_CARD.md               # karta modelu
├── requirements.txt            # zależności

├── raport techniczny.pdf

├── prezentacja nietechniczna.pdf
├── README.md
└── artifacts/                  # tworzone przez notebook przy uruchomieniu
    ├── splits.csv              # podział train/val/test (indeksy wierszy)
    ├── config_wyniki.json      # parametry, metody kalibracji, δ, progi
    ├── metryki_test.csv
    ├── backtest_LR_test.csv, backtest_XGB_test.csv
    ├── decyzje_test.csv
    ├── audyt_podgrup_test.csv
    ├── psi_csi.csv
    └── predykcje_test.csv

```

## Instalacja

Notebook uruchomiono na Pythonie 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Jedyne wersje zapisane w wynikach notebooka to Python 3.12.16 i scikit-learn 1.9.1; pozostałe
pakiety w `requirements.txt` mają dolne ograniczenia. Dokładne środowisko można zamrozić
poleceniem `pip freeze > requirements-lock.txt`.

## Dane

Plik `zbiór_3.csv` musi leżeć w katalogu, z którego uruchamiany jest notebook (ścieżka
`DATA_PATH` w pierwszej komórce). Oczekiwany format: 3000 wierszy, 223 kolumny, target
`default` (0/1, 405 defaultów = 13,5%), 217 cech numerycznych i 5 kategorycznych
(`szczegolnaFormaPrawna_Symbol`, `formaWlasnosci_Symbol`, `pkdKod`, `schemat_wsk_bilans`,
`schemat_wsk_rzis`). Brak w nich identyfikatora podmiotu i ekspozycji (EAD = 1).

## Uruchomienie

```bash
jupyter notebook final.ipynb
```

Uruchom wszystkie komórki od góry (*Run All*). Notebook jest deterministyczny (`SEED = 42`).

**Tryb szybki.** Pierwsza komórka ustawia `os.environ["NB_FAST"] = "0"` (pełne uruchomienie:
40 losowań XGBoost, 20 seedów LIME, 500 replikacji bootstrapu). Do szybkiego sprawdzenia, czy
wszystko działa, zmień tę wartość w pierwszej komórce na `"1"`; wyniki będą wtedy inne niż
w model card.

## Co robi notebook

|Sekcja|Zawartość|
|-|-|
|0–2|Konfiguracja, kontrola jakości danych, podział 60/20/20 + 5-fold CV|
|3|EDA na train, kontrola przecieku (AUC jednowymiarowe)|
|4|Pipeline cech (inf → NaN, winsoryzacja, imputacja + flagi braków, filtry korelacji/VIF, One-Hot) – wszystko dopasowywane wewnątrz foldów|
|5|Regresja logistyczna (L1), SHAP dla modelu liniowego|
|6|XGBoost, TreeSHAP, ważność permutacyjna, studia przypadków, PDP/ICE/ALE|
|7|LIME, wierność i stabilność wyjaśnień, surogat globalny|
|8|Kalibracja: kształt (Platt/izotoniczna/beta) + przesunięcie do średniego PD = 4%|
|9|Progi decyzji, skala ratingowa (7 klas), backtest, kontrfaktyk|
|10|Ocena końcowa na teście (użytym dopiero tutaj), audyt podgrup|
|11|Monitoring: PSI/CSI|

Zbiór testowy nie bierze udziału w żadnej decyzji (strojenie – CV na train; kalibratory,
δ i progi – val).

## Uwagi praktyczne

* **Ostatnia komórka nadpisuje `MODEL_CARD.md`** automatycznie wygenerowanym szkicem
(`open("MODEL_CARD.md", "w")`). Po ponownym uruchomieniu notebooka przywróć uzupełnioną
wersję z repozytorium (np. `git checkout MODEL_CARD.md`) albo zmień nazwę pliku wyjściowego
w tej komórce.
* `assert DATA_PATH` w pierwszej komórce nie sprawdza, czy plik istnieje; brak
`zbiór_3.csv` objawi się dopiero błędem `FileNotFoundError` przy wczytywaniu danych.
* XGBoost używa `n_jobs=4`; strojenie i CV działają jednowątkowo (`n_jobs=1` w searchach),
więc pełne uruchomienie może trwać dłużej niż tryb szybki.
* Pipeline'y są cache'owane w `artifacts/cache`; po zmianie kodu przetwarzania cech usuń ten
katalog.
* Wyniki mogą się nieznacznie różnić między wersjami bibliotek (zwłaszcza XGBoost i SHAP).

## Ograniczenia

Dane przekrojowe (brak wymiaru czasowego), po 81 defaultów w val i test, PD po kalibracji do 4%
nie jest statystycznym oszacowaniem częstości defaultów w próbie, a wyjaśnienia opisują model,
nie przyczynowość. Pełna lista w [MODEL\_CARD.md](MODEL_CARD.md#ograniczenia-i-ryzyka).

