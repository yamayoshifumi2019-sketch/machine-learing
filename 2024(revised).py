import fastf1
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# 1. キャッシュを有効にする
fastf1.Cache.enable_cache("f1_cache")

# --- 2. 学習フェーズ (2023年データ) ---
print("--- ステップ1: 2023年のデータから学習中... ---")
session_2023_q = fastf1.get_session(2023, 3, "Q")
session_2023_q.load()
session_2023_r = fastf1.get_session(2023, 3, "R")
session_2023_r.load()

# 予選：ベストタイム取得 (Q1, Q2, Q3の最小値)
res_2023 = session_2023_q.results
q_2023 = res_2023[['Abbreviation']].copy()
q_2023['QTime_s'] = res_2023[['Q1', 'Q2', 'Q3']].min(axis=1).dt.total_seconds()

# 決勝：平均ラップタイム算出
laps_2023 = session_2023_r.laps.groupby('Driver')['LapTime'].mean().reset_index()
laps_2023['AvgRaceTime_s'] = laps_2023['LapTime'].dt.total_seconds()

# 学習用データの結合
train_data = pd.merge(q_2023, laps_2023, left_on='Abbreviation', right_on='Driver')
train_data.dropna(inplace=True)

# 学習実行 (X=予選ベスト, y=決勝平均)
X_train = train_data[['QTime_s']]
y_train = train_data['AvgRaceTime_s']
model = GradientBoostingRegressor(n_estimators=100, random_state=39)
model.fit(X_train, y_train)

# --- 3. 予測フェーズ (2024年予選データ) ---
print("\n--- ステップ2: 2024年の結果を予測中... ---")
qualifying_2024 = pd.DataFrame({
    "DriverName": ["Max Verstappen", "Carlos Sainz", "Sergio Perez", "Lando Norris", "Charles Leclerc", 
                   "Oscar Piastri", "George Russell", "Yuki Tsunoda", "Lance Stroll", "Fernando Alonso", 
                   "Lewis Hamilton", "Alexander Albon", "Valtteri Bottas", "Kevin Magnussen", 
                   "Esteban Ocon", "Nico Hulkenberg", "Pierre Gasly", "Daniel Ricciardo", "Zhou Guanyu"],
    "TLA": ["VER", "SAI", "PER", "NOR", "LEC", "PIA", "RUS", "TSU", "STR", "ALO", 
            "HAM", "ALB", "BOT", "MAG", "OCO", "HUL", "GAS", "RIC", "ZHO"],
    "QTime_s": [75.915, 76.185, 76.274, 76.315, 76.435, 76.572, 76.724, 76.788, 
                77.072, 77.552, 76.960, 77.135, 77.340, 77.427, 77.697, 77.976, 
                77.982, 78.085, 78.188]
})

# 2024年の予測ペースを算出
qualifying_2024["PredictedPace"] = model.predict(qualifying_2024[["QTime_s"]])

# --- 4. 答え合わせ (2024年決勝データとの照合) ---
print("--- ステップ3: 実際の結果と比較・検証中... ---")
session_2024_r = fastf1.get_session(2024, 3, "R")
session_2024_r.load()

# 2024年の実際の結果とステータス（完走・リタイア）を取得
res_2024 = session_2024_r.results[['Abbreviation', 'Status']]
laps_2024 = session_2024_r.laps.groupby('Driver')['LapTime'].mean().reset_index()
laps_2024['ActualPace'] = laps_2024['LapTime'].dt.total_seconds()

# 予測・実績・ステータスをすべて結合
comparison = pd.merge(qualifying_2024, laps_2024, left_on='TLA', right_on='Driver')
comparison = pd.merge(comparison, res_2024, left_on='TLA', right_on='Abbreviation')

# 【重要】完走者（Finished または 周回遅れ完走）だけに絞り込む
# これによりリタイアしたVER, HAM, RUSなどが除外され、純粋なペース予測精度が測れます
comparison = comparison[comparison['Status'].str.contains('Finished|Lap', na=False)]

# --- 5. 結果表示 ---
comparison = comparison.sort_values(by="PredictedPace").reset_index(drop=True)
comparison.index += 1

print("\n📊 --- 2024年オーストラリアGP 予測 vs 実績（完走者のみ） --- 📊")
print(comparison[["DriverName", "PredictedPace", "ActualPace"]])

# 平均絶対誤差 (MAE) の計算
mae = mean_absolute_error(comparison['ActualPace'], comparison['PredictedPace'])

print(f"\n🔍 AIの予測精度 (MAE): {mae:.2f} 秒")
