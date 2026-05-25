# ==============================
# INSTALLS
# ==============================
# !pip install numpy pandas pywt scikit-learn xgboost


# ==============================
# IMPORTS
# ==============================
import os
import numpy as np
import pandas as pd
import pywt

from scipy.signal import ellip, filtfilt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample, shuffle

from sklearn.model_selection import KFold, LeaveOneGroupOut

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# ==============================
# CONSTANTS
# ==============================
DATASET_PATH = "dataset-processed"

FS = 200
EPOCH_SECONDS = 8
EPOCH_SAMPLES = FS * EPOCH_SECONDS

LABEL_MAP = {"C": 0, "F": 1, "A": 1}

FEATURE_TYPES = ["LBP", "AE", "RMS"]


# ==============================
# FILTER
# ==============================
def bandpass_filter(signal):
    nyq = 0.5 * FS
    low = 0.1 / nyq
    high = 60 / nyq

    b, a = ellip(4, 1, 40, [low, high], btype="band")
    return filtfilt(b, a, signal)


# ==============================
# FEATURE EXTRACTION
# ==============================
def extract_feature(epoch, feature_type):

    coeffs = pywt.wavedec(epoch, "db4", level=4)
    features = []

    for c in coeffs:

        if feature_type == "LBP":
            val = np.log(np.mean(c**2) + 1e-10)

        elif feature_type == "AE":
            val = np.sum(c**2)

        elif feature_type == "RMS":
            val = np.sqrt(np.mean(c**2))

        features.append(val)

    return features


# ==============================
# DATASET BUILDER
# ==============================
def build_dataset(feature_type):

    X, y, groups = [], [], []

    subjects = sorted(os.listdir(DATASET_PATH))

    for subject in subjects:

        subject_path = os.path.join(DATASET_PATH, subject)

        if not os.path.isdir(subject_path):
            continue

        exports = os.path.join(subject_path, "exports")

        eeg_file, meta_file = None, None

        for f in os.listdir(exports):
            if "channels_timeseries" in f:
                eeg_file = f
            if "metadata" in f:
                meta_file = f

        if eeg_file is None or meta_file is None:
            continue

        eeg_path = os.path.join(exports, eeg_file)
        meta_path = os.path.join(exports, meta_file)

        eeg = pd.read_csv(eeg_path).values.flatten()
        meta = pd.read_csv(meta_path)

        eeg = bandpass_filter(eeg)

        for i in range(0, len(eeg) - EPOCH_SAMPLES, EPOCH_SAMPLES):

            epoch = eeg[i:i + EPOCH_SAMPLES]

            label_char = meta.iloc[i]["label"] if "label" in meta.columns else "C"
            label = LABEL_MAP.get(label_char, 0)

            feat = extract_feature(epoch, feature_type)

            X.append(feat)
            y.append(label)
            groups.append(subject)

    return np.array(X), np.array(y), np.array(groups)


# ==============================
# BALANCING
# ==============================
def balance_data(X, y):

    df = pd.DataFrame(X)
    df["label"] = y

    class0 = df[df.label == 0]
    class1 = df[df.label == 1]

    if len(class0) < len(class1):

        class1_down = resample(class1, replace=False,
                               n_samples=len(class0), random_state=42)
        df_bal = pd.concat([class0, class1_down])

    else:

        class0_down = resample(class0, replace=False,
                               n_samples=len(class1), random_state=42)
        df_bal = pd.concat([class0_down, class1])

    y_bal = df_bal["label"].values
    X_bal = df_bal.drop(columns=["label"]).values

    return X_bal, y_bal


# ==============================
# MODELS
# ==============================
models = {
    "SVM": SVC(kernel="rbf", class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced"),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss"
    )
}


# ==============================
# K-FOLD TRAINING (UPDATED)
# ==============================
def train_model_kfold(name, model, X, y):

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # balance
        X_train, y_train = balance_data(X_train, y_train)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)

        print(f"{name} Fold {fold}: {acc:.4f}")

    mean_acc = np.mean(scores)
    std_acc = np.std(scores)

    print(f"{name} Mean: {mean_acc:.4f} | Std: {std_acc:.4f}")
    print(f"{name} All Scores: {scores}\n")

    return mean_acc, std_acc, scores


# ==============================
# RUN K-FOLD
# ==============================
results = []

for feature in FEATURE_TYPES:

    print("\n============================")
    print("Feature:", feature)
    print("============================")

    X, y, groups = build_dataset(feature)
    y = y.astype(int)

    X, y, groups = shuffle(X, y, groups, random_state=42)

    for name, model in models.items():

        mean_acc, std_acc, scores = train_model_kfold(name, model, X, y)

        results.append({
            "Feature": feature,
            "Model": name,
            "Accuracy": mean_acc,
            "Std": std_acc,
            "All_Scores": scores
        })


results_df = pd.DataFrame(results)

print("\n10-Fold Cross Validation Results")
print(results_df)


# ==============================
# LOSO TRAINING
# ==============================
def train_model_loso(name, model, X, y, groups):

    logo = LeaveOneGroupOut()
    scores = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), 1):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train, y_train = balance_data(X_train, y_train)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)

        print(f"{name} LOSO Fold {fold}: {acc:.4f}")

    mean_acc = np.mean(scores)
    std_acc = np.std(scores)

    print(f"{name} LOSO Mean: {mean_acc:.4f} | Std: {std_acc:.4f}")
    print(f"{name} LOSO Scores: {scores}\n")

    return mean_acc, std_acc


# ==============================
# RUN LOSO
# ==============================
loso_results = []

for feature in FEATURE_TYPES:

    print("\n============================")
    print("LOSO Feature:", feature)
    print("============================")

    X, y, groups = build_dataset(feature)
    y = y.astype(int)

    for name, model in models.items():

        mean_acc, std_acc = train_model_loso(name, model, X, y, groups)

        loso_results.append({
            "Feature": feature,
            "Model": name,
            "Accuracy": mean_acc,
            "Std": std_acc
        })


loso_df = pd.DataFrame(loso_results)

print("\nLOSO Results")
print(loso_df)