import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestClassifier,
                               VotingClassifier,
                               GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("  PhishGuard Pro — Model Training")
print("=" * 55)

# ── Load or generate dataset ──────────────────────────────────
print("\n[1/5] Loading dataset...")
try:
    try:
        from scipy.io import arff
        data, meta = arff.loadarff('Training Dataset.arff')
        df = pd.DataFrame(data).apply(lambda x: x.map(lambda v: int(v)))
        print(f"  Loaded UCI ARFF dataset: {df.shape}")
    except:
        df = pd.read_csv('dataset.csv')
        print(f"  Loaded CSV dataset: {df.shape}")
except:
    print("  Generating synthetic dataset (11,055 samples)...")
    np.random.seed(42)
    n = 11055
    phish = np.zeros(n, dtype=bool)
    phish[:int(n * 0.55)] = True
    np.random.shuffle(phish)

    def gf(pv, lv, noise=0.15):
        v = np.where(phish, pv, lv)
        idx = np.random.choice(n, int(n * noise), replace=False)
        v[idx] = np.random.choice([-1, 0, 1], len(idx))
        return v

    cols = [
        'having_IP_Address','URL_Length','Shortining_Service',
        'having_At_Symbol','double_slash_redirecting','Prefix_Suffix',
        'having_Sub_Domain','SSLfinal_State','Domain_registeration_length',
        'Favicon','port','HTTPS_token','Request_URL','URL_of_Anchor',
        'Links_in_tags','SFH','Submitting_to_email','Abnormal_URL',
        'Redirect','on_mouseover','RightClick','popUpWidnow','Iframe',
        'age_of_domain','DNSRecord','web_traffic','Page_Rank',
        'Google_Index','Links_pointing_to_page','Statistical_report'
    ]
    pv = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    lv = [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
    data = {c: gf(p, l) for c, p, l in zip(cols, pv, lv)}
    data['Result'] = np.where(phish, -1, 1)
    df = pd.DataFrame(data)
    df.to_csv('dataset.csv', index=False)
    print(f"  Generated: {df.shape}")

# ── Preprocess ────────────────────────────────────────────────
print("\n[2/5] Preprocessing...")
feature_cols = [c for c in df.columns if c != 'Result']
X = df[feature_cols].values
y = np.where(df['Result'].values == -1, 1, 0)
print(f"  Total: {len(y)} | Phishing: {y.sum()} | Legit: {(1-y).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── Train ─────────────────────────────────────────────────────
print("\n[3/5] Training ensemble (RF + GBM + LR)...")
rf  = RandomForestClassifier(n_estimators=200, max_depth=15,
                              random_state=42, n_jobs=-1)
gbm = GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                  random_state=42)
lr  = LogisticRegression(max_iter=1000, random_state=42)

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gbm', gbm), ('lr', lr)],
    voting='soft'
)
ensemble.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────
print("\n[4/5] Evaluating...")
y_pred = ensemble.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
cv     = cross_val_score(ensemble, X, y, cv=5, scoring='accuracy')

rf.fit(X_train, y_train)
gbm.fit(X_train, y_train)
lr.fit(X_train, y_train)

comparison = {
    'Logistic Regression': accuracy_score(y_test, lr.predict(X_test)),
    'Random Forest':       accuracy_score(y_test, rf.predict(X_test)),
    'Gradient Boosting':   accuracy_score(y_test, gbm.predict(X_test)),
    'Ensemble (Ours)':     acc,
}
fi = pd.DataFrame({
    'feature':    feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n  Test Accuracy : {acc*100:.2f}%")
print(f"  CV  Accuracy  : {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Legitimate','Phishing']))
print("\n  Model Comparison:")
for name, a in comparison.items():
    print(f"    {name:25s}: {a*100:.2f}%")
print("\n  Top 5 Features:")
for _, row in fi.head(5).iterrows():
    print(f"    {row['feature']:35s}: {row['importance']:.4f}")

# ── Save ──────────────────────────────────────────────────────
print("\n[5/5] Saving model...")
with open('phishguard_model.pkl', 'wb') as f:
    pickle.dump({
        'ensemble':    ensemble,
        'rf':          rf,
        'feature_cols':feature_cols,
        'accuracy':    acc,
        'cv':          cv,
        'comparison':  comparison,
        'fi':          fi,
    }, f)

print("  Saved: phishguard_model.pkl")
print(f"\n{'='*55}")
print(f"  Training complete! Accuracy: {acc*100:.2f}%")
print(f"{'='*55}")
print("\n  Now run:  streamlit run app.py")