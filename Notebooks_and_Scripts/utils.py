import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
import lightgbm as lgbm
from lightgbm import LGBMRegressor
import optuna
import gc
import warnings
import traceback
from collections import defaultdict
import math

KELVIN_THRESHOLD = 100
TARGET_COLS = ["Avg_Temperature", "Radiation", "Rain_Amount", "Wind_Speed", "Wind_Direction"]
N_SPLITS_FEAT_SELECT = 5


def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred); numerator = np.abs(y_pred - y_true); denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = np.where(denominator < 1e-9, 0, numerator / denominator); return np.mean(ratio) * 100

def deg_to_sin(degrees): return np.sin(np.radians(degrees))
def deg_to_cos(degrees): return np.cos(np.radians(degrees))
def sincos_to_deg(sin_val, cos_val): return np.degrees(np.arctan2(sin_val, cos_val)) % 360

def convert_units(df):
    df = df.copy(); temp_cols = ["Avg_Temperature", "Avg_Feels_Like_Temperature"]; print("Converting units...")

    for col in temp_cols:

        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            valid_temps = df[col].dropna()
            if not valid_temps.empty and valid_temps.max() > KELVIN_THRESHOLD: print(f"  Converting {col}..."); df[col] = df[col].apply(lambda x: x - 273.15 if pd.notna(x) and x > KELVIN_THRESHOLD else x)
    
    return df

def create_geo_clusters(train_df, test_df, n_clusters, seed):
    print("\nCreating geo clusters..."); train_df['geo_cluster'] = -1; test_df['geo_cluster'] = -1

    if 'latitude' in train_df.columns and 'longitude' in train_df.columns:
        coords_df = train_df[["latitude", "longitude"]].dropna().drop_duplicates()

        if len(coords_df) >= n_clusters:
            try:
                try: kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto').fit(coords_df)
                except ValueError: kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(coords_df)

                coord_to_cluster = {tuple(row): kmeans.labels_[i] for i, row in enumerate(coords_df.itertuples(index=False))}

                def get_cluster(row):
                    if pd.isna(row['latitude']) or pd.isna(row['longitude']): return -1
                    return coord_to_cluster.get((row['latitude'], row['longitude']), -1)
                
                train_df['geo_cluster'] = train_df.apply(get_cluster, axis=1)

                kingdom_modes = train_df[train_df['geo_cluster'] != -1].groupby('kingdom')['geo_cluster'].agg(lambda x: x.mode()[0] if not x.mode().empty else -1)
                kingdom_cluster_map = kingdom_modes.to_dict(); test_df['geo_cluster'] = test_df['kingdom'].map(kingdom_cluster_map).fillna(-1)

                print(f"  Geo-clustering successful: {n_clusters} clusters.")

            except Exception as e: print(f"  Error KMeans: {e}. Defaulting geo_cluster.")

        else: print(f"  Not enough unique coords ({len(coords_df)}). Defaulting geo_cluster.")

    else: print("  Lat/Lon not found. Defaulting geo_cluster.")

    return train_df, test_df

def create_time_features(df_input):

    df = df_input.copy(); df_name = "Training" if "Avg_Temperature" in df.columns else "Test"

    print(f"\n--- Processing time features for {df_name} (Shape: {df.shape}) ---")

    original_rows = len(df); date_cols = ['Year', 'Month', 'Day']
    if not all(col in df.columns for col in date_cols): print(f"Error: Missing date columns."); return pd.DataFrame()

    print("  1: Convert Y/M/D numeric..."); df[date_cols] = df[date_cols].apply(pd.to_numeric, errors='coerce')

    rows_with_nans = df[date_cols].isnull().any(axis=1); num_rows_with_nans = rows_with_nans.sum()

    if num_rows_with_nans > 0: print(f"  2: Dropping {num_rows_with_nans} non-numeric rows."); df = df.dropna(subset=date_cols);
    else: print("  2: No non-numeric Y/M/D rows.")

    if df.empty: print(f"  Error: {df_name} empty after step 2."); return df

    print("  2a: Adjusting 'Year' +2000..."); df['Year'] = df['Year'] + 2000
    print(f"  3: Date conversion (Example Year: {df['Year'].iloc[0] if not df.empty else 'N/A'})...")

    try: df[date_cols] = df[date_cols].astype(int); df['date'] = pd.to_datetime(df[date_cols], errors='coerce')
    except Exception as e: print(f"  CRITICAL Error pd.to_datetime: {e}"); traceback.print_exc(); return pd.DataFrame()

    nat_rows = df['date'].isna().sum()

    if nat_rows > 0: print(f"  4: Dropping {nat_rows} invalid date combos (NaT)."); df = df.dropna(subset=['date']);
    else: print("  4: All adjusted Y/M/D valid.")

    if df.empty: print(f"  Error: {df_name} empty after step 4."); return df

    final_rows = len(df); print(f"  Dates created for {final_rows}/{original_rows} rows.")

    print("  5: Creating derived time features...");

    try:
        df['dayofyear'] = df['date'].dt.dayofyear; df['dayofweek'] = df['date'].dt.dayofweek; df['month'] = df['date'].dt.month
        df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int); df['year'] = df['date'].dt.year; df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int); df['days_in_month'] = df['date'].dt.days_in_month
        df['dayofyear_sin'] = np.sin(2*np.pi*df['dayofyear']/366); df['dayofyear_cos'] = np.cos(2*np.pi*df['dayofyear']/366)
        df['month_sin'] = np.sin(2*np.pi*df['month']/12); df['month_cos'] = np.cos(2*np.pi*df['month']/12)
        df['dayofweek_sin'] = np.sin(2*np.pi*df['dayofweek']/7); df['dayofweek_cos'] = np.cos(2*np.pi*df['dayofweek']/7)
        df['quarter_sin'] = np.sin(2*np.pi*df['quarter']/4); df['quarter_cos'] = np.cos(2*np.pi*df['quarter']/4)
        df['month_x_dayofyear_sin'] = df['month'] * df['dayofyear_sin']; df['week_x_dayofweek_cos'] = df['weekofyear'] * df['dayofweek_cos']; df['day_x_month_sin'] = df['date'].dt.day * df['month_sin']

    except Exception as e: print(f"  Error creating derived time features: {e}.")

    print(f"--- Finished time features {df_name}, shape: {df.shape} ---")

    return df

def create_lag_rolling_features_advanced(df, cols_to_process):

    df = df.copy() # Work on a copy

    df_type = 'Train' if any(t in df.columns for t in TARGET_COLS) else 'Test'
    valid_cols = [col for col in cols_to_process if col in df.columns]

    print(f"\nCreating ADVANCED lags/rolling for {df_type} (Shape: {df.shape}) using: {valid_cols}")

    if df.empty or not valid_cols: print(f"  Skipping."); return df

    print("  Ensuring numeric..."); df[valid_cols] = df[valid_cols].apply(pd.to_numeric, errors='coerce', downcast='float')
    print("  Calculating features...")

    groupby_col = "kingdom"
    new_feature_data = {} # Store new features here temporarily

    for col in valid_cols:
        print(f"    Processing column: {col}")

        if df[col].isnull().all(): print(f"      Skipping '{col}' (all NaN)."); continue

        grouped = df.groupby(groupby_col)[col] 

        for lag in [1, 2, 3, 5, 7, 14, 30]: new_feature_data[f"{col}_lag{lag}"] = grouped.shift(lag)

        # Rolling Stats & Quantiles
        for window in [3, 7, 14, 30, 60]:
             print(f"      Rolling window: {window}...")

             shifted = grouped.shift(1)

             base_roll = shifted.rolling(window, min_periods=1)
             new_feature_data[f"{col}_roll_mean{window}"] = base_roll.mean()
             new_feature_data[f"{col}_roll_std{window}"] = base_roll.std()
             new_feature_data[f"{col}_roll_median{window}"] = base_roll.median()
             new_feature_data[f"{col}_roll_min{window}"] = base_roll.min()
             new_feature_data[f"{col}_roll_max{window}"] = base_roll.max()
             new_feature_data[f"{col}_roll_q25_{window}"] = base_roll.quantile(0.25)
             new_feature_data[f"{col}_roll_q75_{window}"] = base_roll.quantile(0.75)
             new_feature_data[f"{col}_roll_skew{window}"] = base_roll.skew()
             del base_roll, shifted; gc.collect() # Manual cleanup

        print("      Calculating EWM...")
        for span in [7, 14, 30, 60]: new_feature_data[f'{col}_ewm_span{span}'] = grouped.transform(lambda x: x.ewm(span=span, adjust=False).mean())

        print("      Calculating Diffs...")
        for diff_n in [1, 3, 7, 14]: new_feature_data[f'{col}_diff{diff_n}'] = grouped.diff(diff_n)


    new_features_df = pd.DataFrame(new_feature_data, index=df.index)

    print("  Calculating Interaction Features...")

    if 'Avg_Temperature_lag1' in new_features_df.columns and 'Radiation_lag1' in new_features_df.columns: new_features_df['temp_X_rad_lag1'] = new_features_df['Avg_Temperature_lag1'] * new_features_df['Radiation_lag1']

    if 'Wind_Speed_lag1' in new_features_df.columns and 'Rain_Amount_lag1' in new_features_df.columns: new_features_df['wind_X_rain_lag1'] = new_features_df['Wind_Speed_lag1'] * new_features_df['Rain_Amount_lag1']

    if 'dayofyear_sin' in df.columns and 'Avg_Temperature_lag1' in new_features_df.columns: new_features_df['temp_X_dayofyear_sin'] = df['dayofyear_sin'] * new_features_df['Avg_Temperature_lag1'] 

    if 'geo_cluster' in df.columns and 'Avg_Temperature_lag1' in new_features_df.columns: new_features_df['cluster_X_temp_lag1'] = df['geo_cluster'].astype(int) * new_features_df['Avg_Temperature_lag1'] 


    # Concatenate new features with original dataframe
    df = pd.concat([df, new_features_df], axis=1)
    del new_features_df; gc.collect()

    print(f"  Filling NaNs for {df_type}...")
    df = df.groupby(groupby_col, group_keys=False).apply(lambda x: x.ffill().bfill(), include_groups=False)

    numeric_cols = df.select_dtypes(include=np.number).columns
    nans_before_final_fill = df[numeric_cols].isnull().sum().sum()

    if nans_before_final_fill > 0: print(f"  Final fill for {nans_before_final_fill} numeric NaNs using 0."); df[numeric_cols] = df[numeric_cols].fillna(0)

    print(f"  Finished ADVANCED lag/roll for {df_type}.")

    float_cols = df.select_dtypes(include='float').columns

    for col in float_cols: df[col] = pd.to_numeric(df[col], downcast='float')

    return df.copy()

def select_features(X, y, features_in, n_top, categorical_list, seed):

    print(f"  Running feature selection for target '{y.name}' ({len(features_in)} features)...")

    if y.isnull().all(): print("    Error: Target variable is all NaNs. Cannot select features."); return []

    lgbm_fs = LGBMRegressor(random_state=seed, n_jobs=-1, importance_type='gain')
    cv = KFold(n_splits=N_SPLITS_FEAT_SELECT, shuffle=True, random_state=seed)
    oof_importances = pd.DataFrame(index=features_in); oof_importances['importance'] = 0

    X_imp = X.copy(); y_imp = y.copy()

    if X_imp.isnull().any().any() or y_imp.isnull().any():
        print("    Imputing NaNs (median) temporarily for FS fit...")

        num_cols = X_imp.select_dtypes(include=np.number).columns

        if len(num_cols) > 0 and X_imp[num_cols].isnull().any().any(): # Check if numeric cols exist before imputing
            imputer_fs = SimpleImputer(strategy='median'); X_imp[num_cols] = imputer_fs.fit_transform(X_imp[num_cols])

        if y_imp.isnull().any(): y_imp = y_imp.fillna(y_imp.median())

    lgbm_cats_fs = [c for c in categorical_list if c in X_imp.columns] or 'auto'
    fold_count = 0

    for fold, (train_idx, _) in enumerate(cv.split(X_imp, y_imp)):

        if len(train_idx) == 0: continue

        X_tr, y_tr = X_imp.iloc[train_idx], y_imp.iloc[train_idx]

        try:
            lgbm_fs.fit(X_tr, y_tr, categorical_feature=lgbm_cats_fs)
            fold_imp = pd.Series(lgbm_fs.feature_importances_, index=features_in)
            oof_importances['importance'] += fold_imp
            fold_count += 1
        except Exception as e: print(f"    Error during FS fit (Fold {fold}): {e}"); continue

    if fold_count == 0: print("    Error: Feature selection failed, no folds completed."); return features_in[:n_top]

    mean_importances = (oof_importances['importance'] / fold_count).sort_values(ascending=False)
    selected = mean_importances[mean_importances > 1e-9].head(n_top).index.tolist()

    if not selected: print("    Warning: No features > 0 importance. Selecting top 10."); selected = mean_importances.head(10).index.tolist()

    print(f"  Selected {len(selected)} features (Max: {n_top}).")

    return selected