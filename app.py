import os, json, io, csv, warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
app.config['SECRET_KEY'] = 'spps-nigeria-2024-secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///spps.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB upload limit

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'warning'

# ── Load ML Model ─────────────────────────────────────────────────
BASE      = os.path.dirname(__file__)
model     = joblib.load(os.path.join(BASE, 'models/rf_model.pkl'))
scaler    = joblib.load(os.path.join(BASE, 'models/scaler.pkl'))
imputer   = joblib.load(os.path.join(BASE, 'models/imputer.pkl'))
le_gender = joblib.load(os.path.join(BASE, 'models/le_gender.pkl'))
with open(os.path.join(BASE, 'models/metadata.json')) as f:
    META = json.load(f)

PERF_COLORS = {'Distinction':'#198754','Credit':'#0d6efd','Pass':'#e8a900','At-Risk':'#dc3545'}
PERF_ICONS  = {'Distinction':'🏆','Credit':'✅','Pass':'📘','At-Risk':'⚠️'}

# ── DB Models ─────────────────────────────────────────────────────
class User(UserMixin, db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email    = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role     = db.Column(db.String(20), default='lecturer')
    fullname = db.Column(db.String(120), default='')
    created  = db.Column(db.DateTime, default=datetime.utcnow)

class Prediction(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    student_id   = db.Column(db.String(30))
    student_name = db.Column(db.String(100))
    age          = db.Column(db.Float)
    gender       = db.Column(db.String(10))
    department   = db.Column(db.String(60))
    attendance   = db.Column(db.Float)
    ca_score     = db.Column(db.Float)
    midsem_score = db.Column(db.Float)
    prev_cgpa    = db.Column(db.Float)
    study_hours  = db.Column(db.Float)
    result       = db.Column(db.String(20))
    confidence   = db.Column(db.Float)
    predicted_by = db.Column(db.String(80))
    timestamp    = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(uid):
    return User.query.get(int(uid))

# ── Prediction Helper ─────────────────────────────────────────────
def make_prediction(data):
    gender_enc = 1 if str(data['gender']).strip() == 'Female' else 0
    dept       = str(data['department']).strip()
    dept_cols  = META['dept_columns']

    row = {
        'Age':              float(data['age']),
        'Gender':           float(gender_enc),
        'Attendance_Rate':  float(data['attendance']),
        'CA_Score':         float(data['ca_score']),
        'MidSem_Score':     float(data['midsem_score']),
        'Prev_CGPA':        float(data['prev_cgpa']),
        'Study_Hours_Week': float(data['study_hours']),
    }
    # Set all dept columns to 0, then set matched one to 1
    for col in dept_cols:
        row[col] = 0
    matched = f'Department_{dept}'
    if matched in dept_cols:
        row[matched] = 1
    # else: unknown dept — all zeros (model handles gracefully)

    ordered = [row.get(f, 0) for f in META['feature_names']]
    X = np.array([ordered])
    X = imputer.transform(X)
    X = scaler.transform(X)
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0].max()
    return pred, round(float(proba) * 100, 1)

# ── Routes ────────────────────────────────────────────────────────
@app.route('/')
def index():
    return redirect(url_for('dashboard') if current_user.is_authenticated else url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    total   = Prediction.query.count()
    at_risk = Prediction.query.filter_by(result='At-Risk').count()
    dist    = Prediction.query.filter_by(result='Distinction').count()
    credit  = Prediction.query.filter_by(result='Credit').count()
    pass_   = Prediction.query.filter_by(result='Pass').count()
    recent  = Prediction.query.order_by(Prediction.timestamp.desc()).limit(10).all()
    return render_template('dashboard.html',
        total=total, at_risk=at_risk, dist=dist, credit=credit, pass_=pass_,
        recent=recent, model_results=META.get('model_results',{}),
        feat_imp=META.get('feature_importances',[])[:6],
        colors=PERF_COLORS, icons=PERF_ICONS)

@app.route('/predict', methods=['GET','POST'])
@login_required
def predict():
    result = None
    if request.method == 'POST':
        data = {k: request.form[k] for k in
                ['age','gender','department','attendance','ca_score','midsem_score','prev_cgpa','study_hours']}
        pred, conf = make_prediction(data)
        rec = Prediction(
            student_id=request.form.get('student_id','N/A'),
            student_name=request.form.get('student_name','N/A'),
            age=data['age'], gender=data['gender'], department=data['department'],
            attendance=data['attendance'], ca_score=data['ca_score'],
            midsem_score=data['midsem_score'], prev_cgpa=data['prev_cgpa'],
            study_hours=data['study_hours'],
            result=pred, confidence=conf, predicted_by=current_user.username
        )
        db.session.add(rec)
        db.session.commit()
        result = {'prediction':pred,'confidence':conf,
                  'color':PERF_COLORS[pred],'icon':PERF_ICONS[pred],
                  'student_name':request.form.get('student_name','Student')}
    return render_template('predict.html', result=result,
                           departments=META['departments'], colors=PERF_COLORS)

@app.route('/batch', methods=['GET','POST'])
@login_required
def batch():
    results   = []
    error_msg = None
    summary   = {}

    if request.method == 'POST':
        file = request.files.get('csvfile')
        if not file or not file.filename.endswith('.csv'):
            flash('Please upload a valid .csv file.', 'danger')
            return redirect(url_for('batch'))

        try:
            df = pd.read_csv(file)
        except Exception as e:
            flash(f'Could not read CSV: {e}', 'danger')
            return redirect(url_for('batch'))

        # Column name mapping — flexible matching
        col_map = {
            'student_id':   ['Student_ID','student_id','ID','id'],
            'student_name': ['Student_Name','student_name','Name','name','StudentName'],
            'age':          ['Age','age','AGE'],
            'gender':       ['Gender','gender','GENDER'],
            'department':   ['Department','department','DEPARTMENT','Dept','dept'],
            'attendance':   ['Attendance_Rate','attendance_rate','Attendance','attendance','ATTENDANCE'],
            'ca_score':     ['CA_Score','ca_score','CA','ca','ContinuousAssessment'],
            'midsem_score': ['MidSem_Score','midsem_score','MidSem','midsem','MidSemester'],
            'prev_cgpa':    ['Prev_CGPA','prev_cgpa','CGPA','cgpa','PreviousCGPA'],
            'study_hours':  ['Study_Hours_Week','study_hours_week','StudyHours','study_hours','StudyHoursPerWeek'],
        }

        def find_col(df, candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        mapped = {k: find_col(df, v) for k, v in col_map.items()}
        missing = [k for k, v in mapped.items() if v is None and k not in ('student_id','student_name')]

        if missing:
            flash(f'Missing required columns: {", ".join(missing)}. Please check your CSV headers.', 'danger')
            return redirect(url_for('batch'))

        # Process in chunks to handle large files
        CHUNK = 500
        total_rows = len(df)
        processed  = 0
        errors     = 0

        for i in range(0, total_rows, CHUNK):
            chunk = df.iloc[i:i+CHUNK]
            recs_to_add = []
            for _, row in chunk.iterrows():
                try:
                    data = {
                        'age':         row[mapped['age']],
                        'gender':      row[mapped['gender']],
                        'department':  row[mapped['department']],
                        'attendance':  row[mapped['attendance']],
                        'ca_score':    row[mapped['ca_score']],
                        'midsem_score':row[mapped['midsem_score']],
                        'prev_cgpa':   row[mapped['prev_cgpa']],
                        'study_hours': row[mapped['study_hours']],
                    }
                    pred, conf = make_prediction(data)
                    sid  = str(row[mapped['student_id']]) if mapped['student_id'] else 'N/A'
                    name = str(row[mapped['student_name']]) if mapped['student_name'] else 'N/A'

                    recs_to_add.append(Prediction(
                        student_id=sid, student_name=name,
                        age=data['age'], gender=data['gender'], department=data['department'],
                        attendance=data['attendance'], ca_score=data['ca_score'],
                        midsem_score=data['midsem_score'], prev_cgpa=data['prev_cgpa'],
                        study_hours=data['study_hours'],
                        result=pred, confidence=conf, predicted_by=current_user.username
                    ))
                    results.append({'id':sid,'name':name,'dept':str(data['department']),
                                    'prediction':pred,'confidence':conf,
                                    'color':PERF_COLORS[pred],'icon':PERF_ICONS[pred]})
                    processed += 1
                except Exception:
                    errors += 1
                    continue

            db.session.bulk_save_objects(recs_to_add)
            db.session.commit()

        # Summary
        from collections import Counter
        dist_count = Counter(r['prediction'] for r in results)
        summary = {
            'total':       processed,
            'errors':      errors,
            'distinction': dist_count.get('Distinction',0),
            'credit':      dist_count.get('Credit',0),
            'pass_':       dist_count.get('Pass',0),
            'at_risk':     dist_count.get('At-Risk',0),
        }
        flash(f'✅ {processed} predictions completed successfully! {errors} rows skipped due to errors.', 'success')

    return render_template('batch.html', results=results, summary=summary,
                           colors=PERF_COLORS, icons=PERF_ICONS)

@app.route('/records')
@login_required
def records():
    page   = request.args.get('page', 1, type=int)
    filter_ = request.args.get('filter','')
    dept_f  = request.args.get('dept','')
    q = Prediction.query.order_by(Prediction.timestamp.desc())
    if filter_: q = q.filter_by(result=filter_)
    if dept_f:  q = q.filter_by(department=dept_f)
    preds = q.paginate(page=page, per_page=25)
    all_depts = [d[0] for d in db.session.query(Prediction.department).distinct().all()]
    return render_template('records.html', preds=preds, filter_result=filter_,
                           dept_filter=dept_f, all_depts=sorted(all_depts),
                           colors=PERF_COLORS, icons=PERF_ICONS)

@app.route('/at-risk')
@login_required
def at_risk():
    students = Prediction.query.filter_by(result='At-Risk').order_by(Prediction.timestamp.desc()).all()
    return render_template('at_risk.html', students=students)

@app.route('/analytics')
@login_required
def analytics():
    all_preds = Prediction.query.all()
    from collections import Counter
    dist_count  = Counter(p.result for p in all_preds)
    dept_data   = {}
    for p in all_preds:
        dept_data.setdefault(p.department, {'Distinction':0,'Credit':0,'Pass':0,'At-Risk':0})
        dept_data[p.department][p.result] += 1
    return render_template('analytics.html',
        dist=dist_count.get('Distinction',0), credit=dist_count.get('Credit',0),
        pass_=dist_count.get('Pass',0), risk=dist_count.get('At-Risk',0),
        dept_data=json.dumps(dept_data),
        model_results=META.get('model_results',{}),
        feat_imp=META.get('feature_importances',[]),
        colors=PERF_COLORS)

@app.route('/export/csv')
@login_required
def export_csv():
    preds = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    si = io.StringIO()
    w  = csv.writer(si)
    w.writerow(['#','Student ID','Student Name','Age','Gender','Department',
                'Attendance%','CA Score','MidSem Score','Prev CGPA','Study Hrs/Wk',
                'Prediction','Confidence%','Predicted By','Date'])
    for p in preds:
        w.writerow([p.id, p.student_id, p.student_name, p.age, p.gender, p.department,
                    p.attendance, p.ca_score, p.midsem_score, p.prev_cgpa, p.study_hours,
                    p.result, p.confidence, p.predicted_by,
                    p.timestamp.strftime('%Y-%m-%d %H:%M')])
    out = make_response(si.getvalue())
    out.headers['Content-Disposition'] = 'attachment; filename=spps_predictions.csv'
    out.headers['Content-type'] = 'text/csv'
    return out

@app.route('/admin/users')
@login_required
def admin_users():
    if current_user.role != 'admin':
        flash('Admin access required.', 'danger')
        return redirect(url_for('dashboard'))
    return render_template('admin_users.html', users=User.query.all())

@app.route('/admin/add-user', methods=['POST'])
@login_required
def add_user():
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))
    d = request.form
    if User.query.filter_by(username=d['username']).first():
        flash('Username already exists.', 'danger')
        return redirect(url_for('admin_users'))
    db.session.add(User(username=d['username'], email=d['email'],
        fullname=d['fullname'], role=d['role'],
        password=generate_password_hash(d['password'])))
    db.session.commit()
    flash(f'User {d["username"]} created.', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/delete-user/<int:uid>', methods=['POST'])
@login_required
def delete_user(uid):
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))
    user = User.query.get_or_404(uid)
    if user.id == current_user.id:
        flash('Cannot delete your own account.', 'danger')
        return redirect(url_for('admin_users'))
    db.session.delete(user)
    db.session.commit()
    flash('User deleted.', 'success')
    return redirect(url_for('admin_users'))

# ── DB Init ───────────────────────────────────────────────────────
def init_db():
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='admin').first():
            db.session.add(User(username='admin', email='admin@spps.edu.ng',
                fullname='System Administrator', role='admin',
                password=generate_password_hash('admin123')))
        if not User.query.filter_by(username='lecturer').first():
            db.session.add(User(username='lecturer', email='lecturer@spps.edu.ng',
                fullname='Dr. A. Johnson', role='lecturer',
                password=generate_password_hash('lec123')))
        db.session.commit()
        print("[DB] Ready. Login: admin/admin123 | lecturer/lec123")

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5050)
