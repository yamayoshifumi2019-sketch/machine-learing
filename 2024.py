import fastf1
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# 1. キャッシュを有効にする
fastf1.Cache.enable_cache("f1_cache")

# --- 2. 学習フェーズ (2023年の実績からルールを学ぶ) ---
print("--- ステップ1: 2023年のデータから学習中... ---")
session_2023_q = fastf1.get_session(2023, 3, "Q")
session_2023_q.load()
session_2023_r = fastf1.get_session(2023, 3, "R")
session_2023_r.load()

# 予選：各ドライバーのベストタイムを取得
res_2023 = session_2023_q.results
q_2023 = res_2023[['Abbreviation']].copy()
q_2023['QTime_s'] = res_2023[['Q1', 'Q2', 'Q3']].min(axis=1).dt.total_seconds()

# 決勝：各ドライバーの平均ラップタイムを算出（groupbyを使用）
laps_2023 = session_2023_r.laps.groupby('Driver')['LapTime'].mean().reset_index()
laps_2023['AvgRaceTime_s'] = laps_2023['LapTime'].dt.total_seconds()

# 予選と決勝をマージ（同じドライバーの「原因」と「結果」を横に並べる）
train_data = pd.merge(q_2023, laps_2023, left_on='Abbreviation', right_on='Driver')
train_data.dropna(inplace=True)

# AIモデルの訓練
X_train = train_data[['QTime_s']]     # 学習用のヒント：予選タイム
y_train = train_data['AvgRaceTime_s'] # 学習用の答え：決勝平均タイム
model = GradientBoostingRegressor(n_estimators=100, random_state=39)
model.fit(X_train, y_train)

# --- 3. 予測フェーズ (2024年の予選タイムから未来を予想) ---
print("\n--- ステップ2: 2024年の予選結果をもとに決勝を予測中... ---")
qualifying_2024 = pd.DataFrame({
    "DriverName": ["Max Verstappen", "Carlos Sainz", "Sergio Perez", "Lando Norris", "Charles Leclerc", 
                   "Oscar Piastri", "George Russell", "Yuki Tsunoda", "Lance Stroll", "Fernando Alonso", 
                   "Lewis Hamilton", "Alexander Albon", "Valtteri Bottas", "Kevin Magnussen", 
                   "Esteban Ocon", "Nico Hulkenberg", "Pierre Gasly", "Daniel Ricciardo", "Zhou Guanyu"],
    "QTime_s": [75.915, 76.185, 76.274, 76.315, 76.435, 76.572, 76.724, 76.788, 
                77.072, 77.552, 76.960, 77.135, 77.340, 77.427, 77.697, 77.976, 
                77.982, 78.085, 78.188]
})

# 学習済みモデルを使って予測を実行
qualifying_2024["PredictedRacePace"] = model.predict(qualifying_2024[["QTime_s"]])

# --- 4. 予想結果の表示 ---
# 予測されたペースが速い（秒数が小さい）順に並べ替える
predictions = qualifying_2024.sort_values(by="PredictedRacePace").reset_index(drop=True)
predictions.index += 1 # 1位から表示

print("\n🏁 --- 2024年オーストラリアGP AI予想順位 --- 🏁")
print(predictions[["DriverName", "PredictedRacePace"]])