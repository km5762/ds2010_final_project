import pandas as pd

from pathlib import Path

data_path = Path(__file__).parent / "cleaned_listings.csv"


def main() -> None:
    df = pd.read_csv(data_path)
    print(df.describe())


if __name__ == "__main__":
    assert data_path.exists()
    main()
