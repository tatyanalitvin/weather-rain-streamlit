import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from weather_data_processing import preprocess_data

DATA_PATH = os.environ.get("WEATHER_CSV", "weatherAUS.csv")
MODEL_PATH = "aussie_rain.joblib"


def main():
    print(f"Зчитування даних з {DATA_PATH} ...")
    raw_df = pd.read_csv(DATA_PATH)

    print("Препроцесинг (імпутація, масштабування, кодування) ...")
    data = preprocess_data(raw_df)

    print("Навчання Random Forest ...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        max_features="sqrt",
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(data["X_train"], data["train_targets"])

    train_acc = accuracy_score(
        data["train_targets"], model.predict(data["X_train"])
    )
    val_acc = accuracy_score(data["val_targets"], model.predict(data["X_val"]))
    print(f"Точність на train: {train_acc:.4f}")
    print(f"Точність на val:   {val_acc:.4f}")

    # Зберігаємо модель разом з усіма об'єктами препроцесингу
    aussie_rain = {
        "model": model,
        "imputer": data["imputer"],
        "scaler": data["scaler"],
        "encoder": data["encoder"],
        "input_cols": data["input_cols"],
        "target_col": data["target_col"],
        "numeric_cols": data["numeric_cols"],
        "categorical_cols": data["categorical_cols"],
        "encoded_cols": data["encoded_cols"],
    }

    joblib.dump(aussie_rain, MODEL_PATH, compress=("zlib", 3))
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"Модель збережено у {MODEL_PATH} ({size_mb:.1f} МБ)")


if __name__ == "__main__":
    main()
