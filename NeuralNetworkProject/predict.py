"""
predict.py – Interaktywny kalkulator cen nieruchomości (King County)
─────────────────────────────────────────────────────────────────────
Użycie:
    python predict.py

Wymaga wcześniejszego wytrenowania i zapisu modelu przez pytorch_NN.py
(parametr save_model=True w wywołaniu pytorch_MLP_NN).
"""

import sys
import os

# Dodaj katalog główny projektu do ścieżki, żeby import działał bez instalacji
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms.pytorch_NN import predict_price, MODEL_PATH, SCALER_PATH, FEATURES


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _check_model_exists() -> bool:
    missing = [p for p in (MODEL_PATH, SCALER_PATH) if not os.path.exists(p)]
    if missing:
        print("\n[BŁĄD] Nie znaleziono zapisanego modelu:")
        for p in missing:
            print(f"  • {p}")
        print("\nNajpierw wytrenuj i zapisz model. W main.py wywołaj:")
        print("  pytorch_MLP_NN(ds, ..., save_model=True)")
        return False
    return True


def _get_float(prompt: str, lo: float = None, hi: float = None) -> float:
    while True:
        try:
            val = float(input(prompt))
            if lo is not None and val < lo:
                print(f"  Wartość musi być ≥ {lo}. Spróbuj ponownie.")
                continue
            if hi is not None and val > hi:
                print(f"  Wartość musi być ≤ {hi}. Spróbuj ponownie.")
                continue
            return val
        except ValueError:
            print("  Niepoprawna wartość. Wpisz liczbę.")


def _get_int(prompt: str, lo: int = None, hi: int = None) -> int:
    return int(_get_float(prompt, lo, hi))


# ─── Menu ─────────────────────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════╗
║     PREDYKCJA CENY NIERUCHOMOŚCI – King County       ║
║            Model: PyTorch MLP Regressor              ║
╚══════════════════════════════════════════════════════╝
"""

CURRENT_YEAR = 2026


def prediction_menu() -> None:
    print(BANNER)
    print("Podaj parametry nieruchomości (naciśnij Enter po każdej wartości):\n")

    sqft_living  = _get_float("  Powierzchnia mieszkalna [sqft]  (np. 1500): ", lo=1)
    sqft_lot     = _get_float("  Powierzchnia działki    [sqft]  (np. 5000): ", lo=1)
    lat          = _get_float("  Szerokość geograficzna          (np. 47.50): ", lo=47.0, hi=48.0)
    long_        = _get_float("  Długość geograficzna            (np. -122.2): ", lo=-123.0, hi=-121.0)
    floors       = _get_float("  Liczba pięter   (1 / 1.5 / 2 / 2.5 / 3):  ", lo=1, hi=3.5)
    bedrooms     = _get_int  ("  Liczba sypialni         (np. 3):            ", lo=0, hi=15)
    bathrooms    = _get_float("  Liczba łazienek         (np. 2 lub 1.75):   ", lo=0)
    yr_built     = _get_int  ("  Rok budowy              (np. 1990):          ", lo=1800, hi=CURRENT_YEAR)
    yr_renovated = _get_int  (
        "  Rok ostatniego remontu  (0 = brak remontu):  ",
        lo=0, hi=CURRENT_YEAR
    )

    # Feature engineering – identyczny jak w Dataset
    last_refurb       = yr_renovated if yr_renovated != 0 else yr_built
    years_since_refurb = CURRENT_YEAR - last_refurb
    house_age          = CURRENT_YEAR - yr_built

    features = {
        "sqft_living":       sqft_living,
        "sqft_lot":          min(sqft_lot, 213008),   # clip jak w Dataset
        "lat":               lat,
        "long":              long_,
        "floors":            floors,
        "bedrooms":          bedrooms,
        "bathrooms":         bathrooms,
        "years_since_refurb": years_since_refurb,
        "house_age":          house_age,
    }

    print("\n  Obliczam predykcję...", end=" ", flush=True)
    price = predict_price(features)
    print("gotowe!\n")

    # Wyniki
    print("╔══════════════════════════════════════════════════════╗")
    print(f"║  Szacowana cena nieruchomości:  ${price:>14.0f}  ║")
    print("╚══════════════════════════════════════════════════════╝")

    # Podsumowanie wejść
    print("\n  Parametry wejściowe:")
    print(f"    Powierzchnia mieszkalna : {sqft_living:>8.0f} sqft")
    print(f"    Powierzchnia działki    : {sqft_lot:>8.0f} sqft")
    print(f"    Lokalizacja             : {lat:.4f}°N, {long_:.4f}°E")
    print(f"    Piętra / Sypialnie      : {floors}  /  {bedrooms}")
    print(f"    Łazienki                : {bathrooms}")
    print(f"    Rok budowy              : {yr_built}")
    print(f"    Wiek domu               : {house_age} lat")
    print(f"    Lat od remontu/budowy   : {years_since_refurb} lat")
    print()


def main() -> None:
    if not _check_model_exists():
        sys.exit(1)

    while True:
        prediction_menu()
        again = input("Czy chcesz obliczyć kolejną predykcję? [t/n]: ").strip().lower()
        if again not in ('t', 'tak', 'y', 'yes'):
            print("\nDo widzenia!\n")
            break


if __name__ == "__main__":
    main()
