import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def preprocess_data(raw_df):
    """Готує сирий weatherAUS DataFrame до навчання.

    Повертає словник із матрицями ознак, цілями та навченими об'єктами
    препроцесингу (imputer, scaler, encoder), які потрібні для інференсу.
    """
    # Прибираємо рядки без цільової ознаки або без RainToday
    df = raw_df.dropna(subset=["RainToday", "RainTomorrow"]).copy()

    # Розбиття на train / val за роком
    year = pd.to_datetime(df["Date"]).dt.year
    train_df = df[year < 2015]
    val_df = df[year == 2015]

    input_cols = list(df.columns)[1:-1]  # усе, крім Date та RainTomorrow
    target_col = "RainTomorrow"

    train_inputs = train_df[input_cols].copy()
    train_targets = train_df[target_col].copy()
    val_inputs = val_df[input_cols].copy()
    val_targets = val_df[target_col].copy()

    numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train_inputs.select_dtypes("object").columns.tolist()

    # 1) Імпутація пропущених числових значень середнім
    imputer = SimpleImputer(strategy="mean").fit(df[numeric_cols])
    train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])

    # 2) Масштабування числових ознак у діапазон [0, 1]
    scaler = MinMaxScaler().fit(df[numeric_cols])
    train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])

    # 3) One-hot кодування категоріальних ознак
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(
        df[categorical_cols]
    )
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

    def build_X(inputs):
        encoded = pd.DataFrame(
            encoder.transform(inputs[categorical_cols]),
            columns=encoded_cols,
            index=inputs.index,
        )
        return pd.concat([inputs[numeric_cols], encoded], axis=1)

    X_train = build_X(train_inputs)
    X_val = build_X(val_inputs)

    return {
        "X_train": X_train,
        "train_targets": train_targets,
        "X_val": X_val,
        "val_targets": val_targets,
        "input_cols": input_cols,
        "target_col": target_col,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "encoded_cols": encoded_cols,
        "imputer": imputer,
        "scaler": scaler,
        "encoder": encoder,
    }


def predict_input(model_dict, single_input):
    """Обробляє один введений користувачем запис і повертає (прогноз, ймовірність).

    `single_input` — це dict із сирими значеннями погоди (як у формі додатка).
    Ті самі imputer/scaler/encoder, що й під час навчання, застосовуються до
    нових даних, після чого модель Random Forest робить прогноз.
    """
    numeric_cols = model_dict["numeric_cols"]
    categorical_cols = model_dict["categorical_cols"]
    encoded_cols = model_dict["encoded_cols"]
    imputer = model_dict["imputer"]
    scaler = model_dict["scaler"]
    encoder = model_dict["encoder"]
    model = model_dict["model"]

    input_df = pd.DataFrame([single_input])

    # Той самий препроцесинг, що й на етапі навчання
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    encoded = pd.DataFrame(
        encoder.transform(input_df[categorical_cols]),
        columns=encoded_cols,
        index=input_df.index,
    )
    X = pd.concat([input_df[numeric_cols], encoded], axis=1)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][list(model.classes_).index(pred)]
    return pred, prob
