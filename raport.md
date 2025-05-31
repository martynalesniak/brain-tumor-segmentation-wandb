# Raport z replikacji: Predykcja wieku biologicznego na podstawie MRI mózgu
## 1. Opis zadania
...

## 2. Krótki opis metody z publikacji
...

## 3. Dane i rozkład wieku

...

### Rozkład wieku w naszym zbiorze:
...

## 4. Preprocessing

W artykule, którym się zajmowałyśmy, sposób w jaki wykonany został preprocessing odgrywał kluczową rolę. Celem autorów było jego uproszczenie i udało im się to osiągnąć wykorzystując do preprocessingu wyłącznie jeden krok. Zastosowano tzw. rigid registration obrazów T1-weighted do wzorca MNI (Montreal Neurological Institute template space). Metoda ta polega na wyrównaniu każdego obrazu MRI do standardowego wzorca mózgu, czyli obraz jest jedynie przesuwany i obracany. Znacznie upraszcza to zastosowanie modelu na szerszą skalę, ponieważ preprocessing trwa krótko oraz nie potrzeba bardzo dużych zasobów do jego wykonania. 

W celu wykonania preprocessingu należało wykonać następujące kroki:
- zainstalować pakiet FSL, służący do analizy obrazów MRI. Z pakietu wykorzystano narzędzie FLIRT, które używane jest do liniowej rejestracji obrazów medycznych, czyli dopasowywania jednego obrazu do drugiego, tak aby struktury anatomiczne się pokrywały.
- stworzyć plik .csv w odpowiednim formacie, zawierający między innymi wiek pacjenta, ścieżkę do zdjęcia MRI oraz podział na zbiory testowy, treningowy i walidacyjny. 
- Uruchomić skrypt **`brain_age_trainer_preprocessing.py`**, który wykonuje rigid registration z 6 stopniami swobody wykorzystując narzędzie FLIRT i zapisuje wyniki do określonego folderu. 

Wykonanie rejestracji trwa około jednej minuty na obraz, co jest znaczącym skróceniem czasu w porównaniu do bardziej złożonych procedur preprocessingowych.


## 5. Opis modelu

### Model ResNet3D

W publikacji wykorzystano trójwymiarowy model sieci rezydualnej (**ResNet3D**), zaprojektowany do analizy danych przestrzennych o wymiarach `[wysokość, szerokość, głębokość]`.
Aby zwiększyć dokładność predykcji, autorzy zastosowali **ensemble pięciu niezależnie wytrenowanych modeli ResNet3D**. Ostateczna predykcja była uzyskiwana poprzez uśrednienie wyników wszystkich modeli (średnia arytmetyczna). 

Kążdy z 5 modeli składa się z następujących komponentów:

#### 1. Część konwolucyjna (`ResidualNet3D`)
Zawiera sześć bloków rezydualnych typu **bottleneck**, wykorzystujących warstwy:
- `Conv3D` – konwolucje trójwymiarowe,
- `BatchNorm3D` – normalizacja wsadowa,
- `LeakyReLU` – funkcja aktywacji.

Dodatkowo uwzględniono:
- warstwę początkową z konwolucją 7×7×7 i `MaxPooling3D`,
- końcowe uśrednianie przestrzenne za pomocą `AvgPooling3D`.

#### 2. Bloki rezydualne
Każdy blok wykorzystuje **residual connections**, które umożliwiają dodanie wejścia do wyjścia danego bloku. Mechanizm ten wspiera propagację gradientu, co pozwala efektywnie uczyć głębokie sieci neuronowe.

#### 3. Część w pełni połączona
Po ekstrakcji cech następuje:
- spłaszczenie danych (`flatten`),
- przejście przez trzy warstwy w pełni połączone (`Linear`) z funkcjami aktywacji `ReLU`.

Zadaniem tej części jest **regresja**, tzn. model zwraca pojedynczą wartość liczbową na wyjściu.

## 6. Trening modelu

Ze względu na ograniczone zasoby obliczeniowe dostępne w środowisku Colab, trening modelu przeprowadzono etapowo, w kilku partiach, zamiast jednorazowo:

1. **Podział treningu na partie:**
   - Trening odbywał się w trzech fazach:
     - **Pierwsza faza:** 5 epok
     - **Druga faza:** kolejne 5 epok
     - **Trzecia faza:** ostatnie 10 epok
   - Łącznie model był trenowany przez 20 epok, ale z przerwami na załadunek wagi.

2. **Wczytywanie wag między fazami:**
   - Po zakończeniu każdej fazy treningu:
     - Zapisane zostały finalne wagi modelu (`.pth`).
     - Następnie, przed rozpoczęciem kolejnej fazy, wagi z poprzedniego etapu były wczytywane, 
       aby kontynuować trening od momentu, w którym został przerwany.
   - Dzięki temu możliwe było zachowanie ciągłości uczenia pomimo restartów sesji lub ograniczeń pamięci.


Skrypt  **`brain_age_trainer_holdout.py`** służy do trenowania konwolucyjnej sieci neuronowej (ResNet3D).

#### Główne etapy treningu:

1. **Wczytanie i przygotowanie danych:**
   - Dane wczytywane są z pliku CSV, który zawiera ścieżki do przetworzonych i zarejestrowanych obrazów oraz informacje o wieku i podziale na zbiory (`train`, `dev`, `test`).
   - Na zbiorze treningowym stosowana jest augmentacja obrazów, na zbiorach walidacyjnych i testowych brak augmentacji.

2. **Tworzenie zestawów danych i loaderów:**
   - Tworzone są zestawy danych dla treningu, walidacji (dev) i opcjonalnie testów.
   - Dane ładowane są wsadowo (`batch_size=20`), z równoległym wczytywaniem.

3. **Inicjalizacja modelu i optymalizatora:**
   - Trenowanych jest 5 modeli ResNet3D jako zespół (ensemble).
   - Używany jest optymalizator SGD z początkową stopą uczenia 0.002, która co 5 epok zmniejszana jest dziesięciokrotnie.
   - Funkcja straty to błąd bezwzględny (MAE).

4. **Trening:**
   - Model uczy się przez określoną liczbę epok (domyślnie 20).
   - W każdej epoce przeprowadzane jest uczenie na zbiorze treningowym oraz ocena na zbiorze walidacyjnym.
   - Po ostatniej epoce, jeśli istnieje zbiór testowy, wykonywana jest ewaluacja na tym zbiorze.

5. **Ewaluacja i monitorowanie:**
   - Podczas treningu i ewaluacji zapisywane są metryki MAE oraz korelacja Pearsona między przewidywanym a rzeczywistym wiekiem.
   - Tworzone są wykresy rozrzutu dla każdej epoki i modelu, zapisywane do TensorBoard.
   - Obliczana jest także średnia predykcja (ensemble) z 5 modeli.

6. **Zapis wyników i modeli:**
   - Po każdej epoce zapisywane są wagi modeli w formacie `.pth`.
   - Wyniki predykcji zapisywane są do plików CSV w katalogu wyników.

### Podsumowanie

- Model jest trenowany na przetworzonych obrazach MRI.
- Wykorzystuje się augmentację do zwiększenia zróżnicowania danych treningowych.
- Stosuje się metodę ensemble 5 modeli ResNet3D, co zwiększa stabilność predykcji.
- Stopa uczenia jest adaptacyjnie zmniejszana co 5 epok.
- Monitorowanie odbywa się przy pomocy TensorBoard, z zapisem metryk i wykresów.

### Zmiany kótre zostały przez nas wprowadzone
- możliowść wczytania wag z poprzedniego treningu
  




## 7. Porównanie wyników

...

## 8. Wnioski

...
