import pandas as pd
import numpy as np

np.random.seed(42)
N = 800

age = np.random.randint(17, 35, N)
gender = np.random.choice(['Male', 'Female'], N)
dept = np.random.choice(['Computer Science', 'Business Admin', 'Electrical Eng', 'Mass Communication', 'Accounting'], N)
attendance = np.clip(np.random.normal(74, 14, N), 20, 100)
prev_cgpa = np.clip(np.random.normal(3.1, 0.8, N), 1.0, 5.0)
ca_score = np.clip(prev_cgpa * 7 + np.random.normal(0, 3, N), 5, 40)
midsem = np.clip(prev_cgpa * 2.5 + np.random.normal(0, 2, N), 0, 20)
study_hrs = np.clip(np.random.normal(14, 6, N), 2, 40)

# Exam score influenced by all factors
exam_score = (
    attendance * 0.18 +
    ca_score * 0.6 +
    midsem * 0.9 +
    prev_cgpa * 4.0 +
    study_hrs * 0.25 +
    np.random.normal(0, 4, N)
)
exam_score = np.clip(exam_score, 5, 60)
total_score = ca_score + midsem + exam_score
total_score = np.clip(total_score / 1.2, 0, 100)  # normalize roughly

def classify(score):
    if score >= 70: return 'Distinction'
    elif score >= 60: return 'Credit'
    elif score >= 45: return 'Pass'
    else: return 'At-Risk'

df = pd.DataFrame({
    'Student_ID': [f'STU{str(i+1).zfill(4)}' for i in range(N)],
    'Age': age,
    'Gender': gender,
    'Department': dept,
    'Attendance_Rate': attendance.round(1),
    'CA_Score': ca_score.round(1),
    'MidSem_Score': midsem.round(1),
    'Prev_CGPA': prev_cgpa.round(2),
    'Study_Hours_Week': study_hrs.round(1),
    'Total_Score': total_score.round(1),
    'Performance_Class': [classify(s) for s in total_score]
})

df.to_csv('/home/claude/student_predict/data/student_records.csv', index=False)
print("Dataset generated:", df.shape)
print(df['Performance_Class'].value_counts())
