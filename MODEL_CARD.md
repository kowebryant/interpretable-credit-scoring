# MODEL CARD – scoring/rating PD (Projekt 1)

|||
|-|-|
|**Autorzy**|Filip Kowalczyk, Paweł Stefańczyk|
|**Kontekst**|Projekt 1 z Interpretowalności i Wyjaśnialności Uczenia Maszynowego|
|**Źródło wyników**|`final.ipynb` (seed = 42, pełne uruchomienie, `FAST = False`)|
|**Wersja / data**|1.0 / 17.11.2026|
|**Status**|Model projektowy (edukacyjny). Nie był walidowany niezależnie i nie jest gotowy do użycia produkcyjnego.|

\---

## Cel i zakres

**Cel.** Oszacowanie prawdopodobieństwa niewypłacalności (PD) firmy na podstawie danych finansowych oraz przypisanie jej klasy ratingowej. Porównano dwa modele: interpretowalną regresję logistyczną (LR) i model black-box (XGBoost) wraz z wyjaśnieniami post-hoc (SHAP, LIME, PDP/ICE/ALE, surogat).

**Przewidziane użycie.** Ranking ryzyka, demonstracja metod interpretowalności, wsparcie (nie automatyzacja) decyzji kredytowych w warunkach projektu.

**Poza zakresem.** Użycie poza populacją danych uczących; traktowanie PD po kalibracji do 4% jako statystycznego oszacowania częstości defaultów (patrz „Kalibracja”); decyzje w pełni automatyczne; wnioskowanie przyczynowe.

\---

## Dane

|||
|-|-|
|Plik|`zbiór\_3.csv`|
|Obserwacje / kolumny|3000 / 223 (222 cechy + target `default`)|
|Defaulty|405 (DR = 13,50%)|
|Cechy|217 numerycznych, 5 kategorycznych (`szczegolnaFormaPrawna\_Symbol`, `formaWlasnosci\_Symbol`, `pkdKod`, `schemat\_wsk\_bilans`, `schemat\_wsk\_rzis`)|
|Duplikaty wierszy|brak; brak kolumny identyfikatora podmiotu i ekspozycji (przyjęto 1 wiersz = 1 podmiot, EAD = 1)|
|Wykluczone cechy|brak. Screening AUC jednowymiarowego (sekcja 3.1): maks. 0,737 (`RP\_amortyzacja`, 95,7% braków), brak cech > 0,85, więc brak podejrzenia przecieku|
|Braki|6 kolumn przepływów pieniężnych ≈ 99% braków, `RP\_amortyzacja` ≈ 96%; `wsk\_plynnosc\_biez\_1` 36% braków w train i 941 wartości ±inf|
|Podział|stratyfikowany 60/20/20: train 1800 (243 defaulty), val 600 (81), test 600 (81); zapis w `artifacts/splits.csv`|
|Walidacja|5-fold stratified CV w train (podział grupowy nie dotyczy – brak ID)|

**Przygotowanie cech** (dopasowywane wyłącznie na train / foldzie treningowym): ±inf → NaN; usunięcie cech z > 50% braków lub stałych (16 cech numerycznych); winsoryzacja 1–99%; imputacja medianą + flaga braku dla cech z ≥ 2% braków; filtr |ρ| > 0,95 i iteracyjny VIF > 5; One-Hot z grupowaniem poziomów rzadszych niż 20 obserwacji; standaryzacja tylko dla LR. Wynik: 222 → 153 kolumny (121 numerycznych + flagi, 32 one-hot).

\---

## Modele

||Regresja logistyczna|XGBoost|
|-|-|-|
|Konfiguracja|L1 (liblinear), C = 0,03|`max\_depth` 4, `n\_estimators` 287, `learning\_rate` 0,036, `min\_child\_weight` 5, `subsample` 0,72, `colsample\_bytree` 0,96, `reg\_lambda` 28,1, `gamma` 0|
|Strojenie|GridSearchCV (5-fold)|RandomizedSearchCV, 40 losowań, 5-fold|
|Złożoność|11 niezerowych β z 148 kolumn|104 z 153 kolumn użyte w drzewach|
|AUC w CV (train)|0,751 ± 0,027|0,774 ± 0,020|
|Ograniczenia monotoniczne|brak (liniowy w logicie)|brak|

Uwagi:

* Dla LR wybrano C = 0,03, choć grid search wskazał C = 0,1 (CV AUC 0,757 vs 0,751). Uzasadnienie z notebooka: mniejsze przeuczenie (AUC train 0,77 vs 0,84) i prostszy model.
* XGBoost wykazuje wyraźną lukę train–CV (AUC 0,95 vs 0,77, różnica 0,18), mimo regularyzacji.
* LR: znaki β zgadzają się z oczekiwanymi ekonomicznie tam, gdzie dało się je jednoznacznie wskazać; 6 z 11 testowanych cech jest silnie monotonicznych w decylach.

\---

## Jakość (test, n = 600, 81 defaultów)

|Metryka|LR|XGB|
|-|-|-|
|AUC|0,720 \[0,669; 0,775]|**0,771** \[0,724; 0,820]|
|Gini|0,440|0,541|
|KS|0,354|0,422|
|PR-AUC|0,322|0,367|

Różnica Gini XGB − LR = **+0,101**, CI95 \[+0,021; +0,176] (sparowany bootstrap) – istotna statystycznie, ale dolna granica jest bliska zera, a test ma tylko 81 defaultów.

Dla porównania, AUC XGB na val wynosiło 0,810 (LR: 0,755), a w CV 0,774. Wynik na val jest więc optymistyczny; wiarygodniejsze są CV i test.

\---

## Kalibracja

Trzy etapy: (1) kształt – metoda wybrana przez 5-fold CV wewnątrz val (log-loss), (2) poziom – przesunięcie log-szans δ tak, aby średnie PD na val = 4%, (3) ocena na teście.

||LR|XGB|
|-|-|-|
|Metoda kształtu|Platt|bez kalibracji|
|δ do 4%|−1,473|−1,421|
|Średnie PD na teście: po kształcie|12,86%|13,07%|
|Średnie PD na teście: po 4%|3,63%|4,07%|
|DR na teście|13,50%|13,50%|
|Brier / log-loss po kształcie|0,107 / 0,357|0,103 / 0,339|
|Brier / log-loss po 4%|0,120 / 0,447|0,116 / 0,419|
|CITL po 4%|+1,54|+1,47|

**Kluczowe zastrzeżenie.** 4% to zewnętrzne założenie biznesowe, a nie wynik estymacji. W próbie rzeczywisty DR to 13,5%, więc PD po δ jest systematycznie zaniżone ok. 3,3-krotnie (DR/PD ≈ 3,3 we wszystkich podgrupach). Kotwica celowo pogarsza Brier, log-loss, ECE i CITL, a nie zmienia AUC/KS (transformacja monotoniczna). PD po δ nadaje się do rankingu i klas ratingowych o zadanych progach, ale nie do szacowania oczekiwanej straty w tej próbie bez ponownej kalibracji do właściwego poziomu. Nachylenie kalibracji XGB na teście wynosi 0,88 (< 1 = PD nieco zbyt skrajne).

\---

## Wyjaśnienia

**Globalne (SHAP, TreeSHAP, log-szanse, val).** Top 5 cech wg średniego |φ|: `wsk\_index\_nowe` (0,50), `wsk\_index\_skr\_zaw` (0,27), `wsk\_struktury\_kapitalu` (0,16), `wsk\_poziom\_kapitalu\_obrotowego\_netto` (0,15), `wsk\_index\_aktywne` (0,13). Wartości SHAP kolumn one-hot i flag braków zsumowano do cechy źródłowej.

**Ważność permutacyjna (val, spadek AUC).** Zgadza się z SHAP co do czołówki (`wsk\_index\_nowe` 0,066; `wsk\_index\_skr\_zaw` 0,037), ale nie dla cech skorelowanych: `wsk\_struktury\_kapitalu` ma wysokie SHAP, a ujemną ważność permutacyjną (−0,008). Korelacja rang permutacja–SHAP: 0,39.

**Zgodność z LR.** Korelacja rang ważności SHAP LR vs XGB 0,34, Jaccard top-10 0,33. Kierunki są zgodne w 4 z 4 przypadków, gdzie LR ma niezerowy współczynnik; pozostałe cechy z top-8 XGB zostały wyzerowane przez L1.

**Kształt zależności (PDP XGBoost).** `wsk\_index\_nowe`: rosnący z nieliniowością; `wsk\_index\_skr\_zaw`: rosnący z nieliniowością; `wsk\_struktury\_kapitalu`: malejący z nieliniowością. ICE pokazuje zróżnicowanie wpływu między firmami, ALE ogranicza wpływ korelacji cech (największa różnica PDP–ALE względem zakresu: `wsk\_struktury\_kapitalu`, 36%). PDP dla cech z brakami zakłada brak flagi NA. Test monotoniczności względem oczekiwanych znaków: 11 z 20 najważniejszych cech ma istotne naruszenia (heurystyka), czyli XGBoost nie respektuje hipotez ekonomicznych bez ograniczeń monotonicznych.

**Interakcje (SHAP).** Najsilniejsze pary to cechy `wsk\_index\_\*` między sobą; dalej `wsk\_index\_nowe` × `wsk\_ebitda\_zobowiazan\_2` i × `wsk\_poziom\_kapitalu\_obrotowego\_netto`.

**Lokalne (waterfall + kody przyczyn, 4 przypadki z val; PD surowe).**

|Przypadek|PD XGB|PD LR|Default|Decyzja w notebooku|
|-|-|-|-|-|
|Wysokie ryzyko (obs. 246)|68,9%|27,8%|1|odrzucenie|
|Niskie ryzyko (obs. 2328)|0,63%|7,4%|0|automatyczna akceptacja|
|Graniczny (obs. 355)|10,7%|17,9%|1|ręczna weryfikacja (strefa buforowa ok. 11–17% PD)|
|Największa rozbieżność modeli (obs. 1089)|3,4%|23,5%|0|ręczna weryfikacja|

Rozbieżności LR vs XGB wynikają z wyzerowania przez L1 buforów bilansowych, które XGBoost uwzględnia. W komunikacji z klientem proponuje się czynniki, na które firma ma wpływ (płynność, kapitał obrotowy, zadłużenie, pokrycie długu EBITDA), a pomija się `pkdKod`, formę prawną, staż i indeksy `wsk\_index\_\*`.

**Wierność i stabilność.**

|Miara|Wynik|
|-|-|
|Addytywność SHAP (maks. błąd)|3,4·10⁻⁶|
|LIME, R² poza próbą (logit), mediana / min|0,58 / 0,53 (w próbie: 0,62)|
|LIME, 20 seedów: Spearman / Jaccard top-5|0,64–0,80 / 0,44–0,77|
|SHAP, bootstrap obserwacji: Spearman / Jaccard top-10|0,999 / 0,87|
|SHAP-XGB, refit na foldach: Spearman / Jaccard top-10|0,65 / 0,45|
|SHAP-LR, refit na foldach: Spearman / Jaccard top-10|0,67 / 0,50|

Wniosek: LIME ma umiarkowaną wierność i jest niestabilny między seedami, więc do produkcji preferowany jest deterministyczny TreeSHAP. Niestabilność rankingu po refitcie wynika głównie ze współliniowości wskaźników, a nie z samej metody.

**Surogat globalny** (drzewo regresyjne na log-szansach XGB, głębokość 3, 8 liści): R² na val = 0,59 (Spearman 0,75); PD w liściach od 2,2% do 42,8%. Najlepszy zbadany surogat (głębokość 8) osiąga R² = 0,73 kosztem czytelności. Surogat służy do komunikacji, nie do decyzji.

**Kontrfaktyk** (obs. val 2412, XGB, BB → BBB): minimalna zmiana to spadek `wsk\_liczba\_dni\_istnienia` z 1950 do 1770 dni, czyli „młodsza” firma dostaje lepszy rating, co jest sprzeczne z oczekiwanym znakiem i jest artefaktem progów drzewa. Wynik opisuje model, nie jest wykonalną rekomendacją.

\---

## Skala i decyzje

**Skala:** 7 klas o stałych granicach, wspólnych dla obu modeli: AAA < 1%, AA < 2%, A < 5%, BBB < 8%, BB < 12%, B < 20%, CCC ≥ 20% (PD po δ); klasa D = zaobserwowany default (ex post).

**Rozkład i backtest na teście (XGB, PD po 4%):**

|Klasa|n|DR|Średnie PD|
|-|-|-|-|
|AAA|178|1,7%|0,56%|
|AA|109|10,1%|1,40%|
|A|165|13,3%|3,31%|
|BBB|62|24,2%|6,38%|
|BB|39|30,8%|9,89%|
|B|33|27,3%|15,3%|
|CCC|14|64,3%|25,4%|

HHI: XGB 0,215, LR 0,293 (rozkład równy: 0,143). DR rośnie monotonicznie z klasą dla LR, a dla XGB jest drobna inwersja BB/B (mała próba). Światła backtestu Jeffreysa: po δ XGB 5 czerwonych i 2 żółte, przed δ 6 zielonych i 1 żółte, więc czerwone światła są efektem kotwicy 4%, a nie kształtu kalibracji. Dla LR: po δ 4 czerwone, 2 żółte, 1 zielone; przed δ 6 zielonych i 1 czerwone (klasa A).

**Progi.** Teoretyczny p\* = MARGIN / (MARGIN + LGD) = 0,10 / (0,10 + 0,45) = **0,182**. Próg empiryczny z val (PD po δ): LR 0,035, XGB 0,050. Jest on niższy od p\* tylko dlatego, że PD po δ jest zaniżone; na PD przed δ próg XGB z val wynosi 0,185, czyli blisko p\*. Progi są wrażliwe na założenia kosztowe (próg XGB od 0,015 do 0,085 w siatce LGD × MARGIN).

**Decyzje na teście przy progach zamrożonych na val** (EAD = 1, zysk w jednostkach ekspozycji):

||Akceptacja|DR przyjętych|Zysk|
|-|-|-|-|
|LR, próg z val|69,0%|9,7%|19,4|
|XGB, próg z val|75,3%|8,0%|**25,4**|
|XGB, próg p\*|97,0%|12,4%|18,6|
|Przyjmij wszystkich|100%|13,5%|15,5|

Zgodność decyzji LR vs XGB: 81,7% (110 wniosków). Macierz odrzuceń XGB: TP 45, FP 103, FN 36, TN 416.

**Audyt podgrup (test, XGB):** AUC 0,81 dla SFJIN (n = 310), 0,78 dla SFJMA (n = 131), ale tylko 0,65 dla SFJMI (n = 159, wg notebooka mikrojednostki z uproszczoną sprawozdawczością). Wg aktywów: niski tercyl 0,64, średni 0,81, wysoki 0,84. Wg stażu: 0,73 / 0,78 / 0,78. Podgrupy form własności (poza kodem 214) mają n < 60 i są niewiarygodne. Brak cech chronionych (podmioty gospodarcze).

\---

## Ograniczenia i ryzyka

* **Dane przekrojowe** – brak wymiaru czasowego, więc stabilność czasowa jest nieznana. Podział val/test jest losowy z tej samej próby, więc niskie PSI nie jest dowodem stabilności w czasie.
* **Mała liczba defaultów** (81 na zbiór) – szerokie przedziały ufności (AUC ±0,05), niepewność kalibracji (szczególnie izotonicznej, która w CV wypadła najgorzej), niewiarygodne podgrupy.
* **Kotwica 4%** – PD po δ jest ok. 3,3× niższe od obserwowanego DR; zakłada niezmienny kształt zależności przy zmianie samego poziomu bazowego.
* **Dominacja cech `wsk\_index\_\*`** – brak prostej interpretacji ekonomicznej, a są to cechy, na które firma nie ma wpływu; wymaga potwierdzenia w słowniku danych.
* **Przeuczenie XGBoost** – luka train–CV 0,18 AUC; brak ograniczeń monotonicznych (11/20 cech z naruszeniami).
* **Wyjaśnienia nie dowodzą przyczynowości**; przy skorelowanych cechach SHAP/LIME rozdzielają wkład arbitralnie; SHAP nie wykrywa przecieku. Kontrfaktyk zmienia wskaźniki pojedynczo i nie przelicza powiązań rachunkowych.
* **Słabsza separacja** dla mikrojednostek (SFJMI) i firm o małych aktywach (AUC 0,64–0,65).
* `szczegolnaFormaPrawna\_Symbol` ma w train jedną wartość (117), więc nie niesie informacji.

\---

## Monitoring

Plan: PSI wyniku i CSI cech (kwartalnie), Gini/KS (dryf jakości), średnie PD vs DR oraz Brier/ECE (dryf kalibracji), DR w klasach (backtest), stabilność top-k cech SHAP. Progi alarmowe z sekcji 11 notebooka: PSI > 0,25 (0,10–0,25 = obserwuj), czerwone światło w ≥ 2 klasach, spadek Gini > 5 p.p.

Uwagi do wdrożenia:

* Tabela PSI w notebooku obejmuje wszystkie 222 surowe cechy, w tym 7 usuniętych przez pipeline (> 50% braków). To one dają PSI w zakresie 0,23–9,6; dla cech używanych przez model PSI na teście nie przekracza ok. 0,07, a PSI wyniku jest poniżej 0,05. Przy monitoringu produkcyjnym należy liczyć CSI tylko dla cech zachowanych.
* Reguła „czerwone światło w ≥ 2 klasach” jest już spełniona na teście (5 czerwonych dla XGB) wyłącznie z powodu kotwicy 4%, więc w obecnej postaci alarmowałaby od pierwszego dnia. Należy ją liczyć na PD przed δ albo względem stanu początkowego.

\---

## Artefakty

`artifacts/`: `splits.csv`, `config\_wyniki.json`, `metryki\_test.csv`, `backtest\_LR\_test.csv`, `backtest\_XGB\_test.csv`, `decyzje\_test.csv`, `audyt\_podgrup\_test.csv`, `psi\_csi.csv`, `predykcje\_test.csv`.

