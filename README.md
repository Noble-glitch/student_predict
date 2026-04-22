# AI-Based Student Performance Prediction System (SPPS)
## Final Year Project — Department of Computer Science

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install flask scikit-learn pandas numpy joblib imbalanced-learn \
            flask-sqlalchemy flask-login reportlab matplotlib seaborn
```

### 2. Generate Dataset & Train Model
```bash
python3 data/generate_data.py     # Generate synthetic student dataset
python3 train_model.py            # Train all ML models (RF, DT, LR, GB)
```

### 3. Run the Application
```bash
python3 app.py
```

### 4. Open in Browser
Visit: **http://localhost:5050**

---

## 🔑 Default Login Credentials

| Role     | Username   | Password  |
|----------|------------|-----------|
| Admin    | `admin`    | `admin123`|
| Lecturer | `lecturer` | `lec123`  |

---

## 📁 Project Structure

```
student_predict/
├── app.py                    # Main Flask application
├── train_model.py            # ML model training script
├── data/
│   ├── generate_data.py      # Synthetic dataset generator
│   ├── student_records.csv   # Training dataset (800 records)
│   └── sample_batch_template.csv  # CSV template for batch upload
├── models/
│   ├── rf_model.pkl          # Trained Random Forest model
│   ├── scaler.pkl            # StandardScaler
│   ├── imputer.pkl           # SimpleImputer
│   ├── le_gender.pkl         # LabelEncoder for gender
│   └── metadata.json         # Model metadata & accuracy results
├── templates/
│   ├── base.html             # Base layout with sidebar navigation
│   ├── login.html            # Login page
│   ├── dashboard.html        # Main dashboard with charts
│   ├── predict.html          # Single student prediction
│   ├── batch.html            # Batch CSV upload & prediction
│   ├── records.html          # All prediction records (paginated)
│   ├── at_risk.html          # At-risk students view
│   ├── analytics.html        # Analytics & model comparison charts
│   └── admin_users.html      # User management (admin only)
├── instance/
│   └── spps.db               # SQLite database (auto-created)
└── README.md
```

---

## 🧠 Machine Learning Models

| Algorithm            | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 83.54%   | 82.1%     | 83.5%  | 82.8%    |
| Decision Tree        | 80.66%   | 79.4%     | 80.7%  | 80.0%    |
| Gradient Boosting    | 85.60%   | 84.3%     | 85.6%  | 84.9%    |
| **Random Forest** ✅  | **85.60%**| **85.1%** | **85.6%**|**85.3%**|

**Best Model: Random Forest Classifier** (200 estimators, max_depth=15)

---

## 🎓 Performance Classification (Nigerian Grading System)

| Class       | Total Score | Colour |
|-------------|-------------|--------|
| Distinction | 70 – 100%   | Green  |
| Credit      | 60 – 69%    | Blue   |
| Pass        | 45 – 59%    | Yellow |
| At-Risk     | Below 45%   | Red    |

---

## 🔮 Key Prediction Features (in order of importance)

1. **CA Score** (35.2%) — Continuous Assessment score (0–40)
2. **Previous CGPA** (25.1%) — Prior semester CGPA (0–5.0)
3. **Mid-Semester Score** (16.8%) — Mid-term test score (0–20)
4. **Attendance Rate** (6.8%) — Class attendance percentage
5. **Study Hours/Week** (6.4%) — Self-reported weekly study hours
6. **Age** (4.6%) — Student age
7. **Gender** (1.6%) — Student gender (encoded)

---

## 📊 System Features

- ✅ Single student prediction with confidence score
- ✅ Batch CSV upload for multiple students at once
- ✅ At-risk student alerts with intervention recommendations
- ✅ Interactive analytics dashboard with charts
- ✅ Model accuracy comparison across 4 algorithms
- ✅ Export all predictions to CSV
- ✅ Role-based access (Admin / Lecturer)
- ✅ User management (Admin only)
- ✅ Pagination on records table
- ✅ Nigerian university grading scale

---

## 🛠️ Technology Stack

- **Backend:** Python 3.12, Flask 3.0, SQLAlchemy
- **ML:** Scikit-learn, Imbalanced-learn (SMOTE)
- **Frontend:** Bootstrap 5, Chart.js, Bootstrap Icons
- **Database:** SQLite (easily upgradeable to MySQL/PostgreSQL)
- **Auth:** Flask-Login, Werkzeug password hashing

---

## 📝 Batch Upload CSV Format

Your CSV file must have these exact column headers:

```
Student_ID, Student_Name, Age, Gender, Department,
Attendance_Rate, CA_Score, MidSem_Score, Prev_CGPA, Study_Hours_Week
```

**Valid Departments:**
- Computer Science
- Business Admin
- Electrical Eng
- Mass Communication
- Accounting

---

*SPPS © 2024 — Department of Computer Science*
