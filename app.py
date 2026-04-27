from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS

import __main__  # [추가 1] 이것을 꼭 임포트 해야 합니다.

app = Flask(__name__)
CORS(app)

# =====================================================================
# [추가 2] 모델 학습 시 사용했던 apply_weight 함수를 아래에 그대로 복사해 넣으세요!
# (아래 내용은 예시일 뿐입니다. 원래 노트북 파일에 있던 코드를 넣으세요.)
def apply_weight(x):
    # [여기에 내용 채우기]
    return x 
# =====================================================================

# [추가 3] pkl 파일이 이 함수를 인식할 수 있도록 강제로 연결해 줍니다.
setattr(__main__, 'apply_weight', apply_weight)


# [중요] 반드시 위에서 apply_weight를 정의하고 세팅한 후에 pkl을 불러와야 합니다!
package = joblib.load('full_ssw_package.pkl')
magnus_models = package['magnus_models']

FEATURES_PHYS = ['release_speed', 'release_spin_rate', 'spin_axis', 'release_pos_x', 'release_pos_z']

def simulate_trajectory(vx, vz):
    x, z = 0, 6
    traj = []
    dt = 0.02

    for t in range(60):
        x += vx * dt
        z += vz * dt - 0.5 * 9.8 * dt**2
        traj.append([x, z, t*dt])

    return traj

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

    return jsonify({
        "no_ssw": traj_no_ssw,
        "with_ssw": traj_with_ssw
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
