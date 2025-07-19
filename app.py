import os, json, random, string, time, io, base64
from datetime import datetime, timedelta, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS
import jwt, requests
from dotenv import load_dotenv

# Stylometry imports
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from style_utils import (
    extract_combined_features, chunk_text,
    plot_pca, get_fingerprint_plot, get_author_fingerprint
)

load_dotenv()
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# --- Configuration ---
JWT_SECRET = os.getenv("JWT_SECRET", "secret")
JWT_ALGORITHM = "HS256"
BREVO_API_KEY = os.getenv("BREVO_API_KEY", "")
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
OTP_FILE = 'otp_store.json'
otp_store = {}

# Load/save OTP store
def load_otp():
    global otp_store
    if os.path.exists(OTP_FILE):
        with open(OTP_FILE,'r') as f:
            data = json.load(f)
            for em,data0 in data.items():
                otp_store[em] = {'otp': data0['otp'], 'expires': datetime.fromisoformat(data0['expires'])}
def save_otp():
    with open(OTP_FILE,'w') as f:
        json.dump({em:{'otp':v['otp'], 'expires':v['expires'].isoformat()} for em,v in otp_store.items()}, f)

load_otp()

# JWT creation
def gen_jwt(email):
    payload = {
        'email': email,
        'iat': time.time(),
        'exp': (datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt(req):
    h = req.headers.get('Authorization','')
    if not h.startswith('Bearer '):
        return None, jsonify({'error':'Unauthorized'}), 401
    try:
        return jwt.decode(h.split()[1], JWT_SECRET, algorithms=[JWT_ALGORITHM]), None, None
    except:
        return None, jsonify({'error':'Invalid or expired token'}), 401

# OTP endpoints
@app.route('/api/request-otp', methods=['POST'])
def request_otp():
    data = request.get_json()
    email = data.get('email')
    if not email: return jsonify({'error':'Email required'}), 400
    otp = ''.join(random.choices(string.digits, k=6))
    otp_store[email] = {'otp':otp, 'expires':datetime.now(timezone.utc)+timedelta(minutes=5)}
    save_otp()
    payload = {"sender":{"email":SENDER_EMAIL}, "to":[{"email":email}],
               "subject":"Your OTP","htmlContent":f"<strong>{otp}</strong>"}
    r = requests.post("https://api.brevo.com/v3/smtp/email",
                      json=payload, headers={"api-key":BREVO_API_KEY})
    if r.status_code in (200,201): return jsonify({'message':'OTP sent'}),200
    return jsonify({'error':'OTP send failed'}),500

@app.route('/api/verify-otp', methods=['POST'])
def verify_otp_route():
    data = request.get_json()
    email, otp = data.get('email'), data.get('otp')
    rec = otp_store.get(email)
    if not rec or datetime.now(timezone.utc)>rec['expires']:
        return jsonify({'error':'OTP expired or invalid'}),400
    if rec['otp'] != otp:
        return jsonify({'error':'Wrong OTP'}),400
    del otp_store[email]; save_otp()
    token = gen_jwt(email)
    return jsonify({'token': token}),200

# Stylometry endpoint
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

@app.route('/api/analyze', methods=['POST'])
def analyze():
    user,err,st = verify_jwt(request)
    if err: return err, st
    f1, f2 = request.files.get('author1'), request.files.get('author2')
    if not f1 or not f2:
        return jsonify({'error':'Files missing'}), 400
    t1 = f1.read().decode('utf-8')
    t2 = f2.read().decode('utf-8')
    C1 = chunk_text(t1); C2 = chunk_text(t2)
    feats, labs = [], []
    for c in C1: feats.append(extract_combined_features(c)); labs.append(0)
    for c in C2: feats.append(extract_combined_features(c)); labs.append(1)
    X, y = np.array(feats), np.array(labs)
    if len(X)==0: return jsonify({'error':'No valid text'}),400

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    acc = round(clf.score(X_te, y_te)*100,2)
    rep = classification_report(y_te, clf.predict(X_te), target_names=["Au1","Au2"])

    pca = plot_pca(X, y)
    fp1 = get_fingerprint_plot(X,y,0)
    fp2 = get_fingerprint_plot(X,y,1)
    vec1 = get_author_fingerprint(X[y==0]).tolist()
    vec2 = get_author_fingerprint(X[y==1]).tolist()

    return jsonify({
        'accuracy':acc, 'report':rep,
        'fingerprint1':vec1, 'fingerprint2':vec2,
        'pca_plot':fig_to_b64(pca),
        'fp1_plot':fig_to_b64(fp1),
        'fp2_plot':fig_to_b64(fp2)
    })

# Serve pages
@app.route('/')
def serve_login():
    return app.send_static_file('login.html')

@app.route('/stylometry')
def serve_tool():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT",5000)), debug=True)
