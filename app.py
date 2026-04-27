from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS
import __main__

app = Flask(__name__)   # ← 반드시 이게 먼저!
CORS(app)

def apply_weight(x):
    return x

setattr(__main__, 'apply_weight', apply_weight)

package = joblib.load('full_ssw_package.pkl')
magnus_models = package['magnus_models']

FEATURES_PHYS = ['release_speed', 'release_spin_rate', 'spin_axis', 'release_pos_x', 'release_pos_z']

@app.route('/')          # ← 여기서부터 라우트
def index():
    return jsonify({"status": "ok", "message": "SSW API is running"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    pt = df['pitch_type'][0]
    m_x, m_z = magnus_models[pt]
    pred_x = m_x.predict(df[FEATURES_PHYS])[0]
    pred_z = m_z.predict(df[FEATURES_PHYS])[0]
    traj_no_ssw = simulate_trajectory(pred_x, pred_z)
    traj_with_ssw = simulate_trajectory(data['pfx_x'], data['pfx_z'])
    return jsonify({"no_ssw": traj_no_ssw, "with_ssw": traj_with_ssw})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
