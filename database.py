"""
=============================================================================
PATIENT HISTORY DATABASE & RECORD REPOSITORY (SQLITE)
MCA Final-Year Capstone Project
=============================================================================
Manages patient clinical records, longitudinal diagnostic results,
vital signs, multimodal scores, and CSV exports.
=============================================================================
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "patient_records.db"


def get_db_connection() -> sqlite3.Connection:
    """Establish connection to the SQLite patient database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create patient_records table if it does not already exist."""
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patient_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT NOT NULL,
                    patient_name TEXT DEFAULT 'Anonymous Patient',
                    patient_age REAL,
                    patient_gender TEXT DEFAULT 'Unspecified',
                    disease_module TEXT NOT NULL,
                    imaging_modality TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    scores_json TEXT,
                    vitals_json TEXT,
                    symptoms_text TEXT,
                    image_file TEXT,
                    gradcam_file TEXT,
                    created_at TEXT NOT NULL
                )
            """)
    finally:
        conn.close()


def insert_patient_record(
    patient_id: str,
    disease_module: str,
    imaging_modality: str,
    prediction: str,
    confidence_score: float,
    scores_dict: Optional[Dict] = None,
    vitals_dict: Optional[Dict] = None,
    symptoms_text: Optional[str] = None,
    image_file: Optional[str] = None,
    gradcam_file: Optional[str] = None,
    patient_name: str = "Anonymous Patient",
    patient_age: Optional[float] = None,
    patient_gender: str = "Unspecified",
) -> int:
    """Insert a new patient diagnosis record and return its row ID."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    scores_json = json.dumps(scores_dict) if scores_dict else "{}"
    vitals_json = json.dumps(vitals_dict) if vitals_dict else "{}"

    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO patient_records (
                    patient_id, patient_name, patient_age, patient_gender,
                    disease_module, imaging_modality, prediction, confidence_score,
                    scores_json, vitals_json, symptoms_text, image_file, gradcam_file, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                patient_id,
                patient_name or "Anonymous Patient",
                patient_age,
                patient_gender or "Unspecified",
                disease_module,
                imaging_modality,
                prediction,
                float(confidence_score),
                scores_json,
                vitals_json,
                symptoms_text or "",
                image_file or "",
                gradcam_file or "",
                now_str,
            ))
            return cursor.lastrowid
    finally:
        conn.close()


def get_all_records(
    search: Optional[str] = None,
    disease_filter: Optional[str] = None,
    limit: int = 150
) -> List[Dict]:
    """Retrieve patient records with optional keyword search and disease filtering."""
    query = "SELECT * FROM patient_records WHERE 1=1"
    params = []

    if disease_filter and disease_filter.strip() != "":
        query += " AND disease_module = ?"
        params.append(disease_filter.strip())

    if search and search.strip() != "":
        s = f"%{search.strip()}%"
        query += " AND (patient_id LIKE ? OR patient_name LIKE ? OR prediction LIKE ?)"
        params.extend([s, s, s])

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_record_by_id(record_id: int) -> Optional[Dict]:
    """Retrieve a single patient record by its database row ID."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM patient_records WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def delete_record(record_id: int) -> bool:
    """Delete a patient record by ID."""
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM patient_records WHERE id = ?", (record_id,))
            return cursor.rowcount > 0
    finally:
        conn.close()


def get_statistics() -> Dict:
    """Calculate summary statistics for dashboard display."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM patient_records")
        total_scans = cursor.fetchone()[0]

        cursor.execute("SELECT disease_module, COUNT(*) FROM patient_records GROUP BY disease_module")
        by_disease = dict(cursor.fetchall())

        cursor.execute("SELECT prediction, COUNT(*) FROM patient_records GROUP BY prediction")
        by_prediction = dict(cursor.fetchall())

        return {
            "total_scans": total_scans,
            "by_disease": by_disease,
            "by_prediction": by_prediction,
        }
    finally:
        conn.close()


# Initialize table on first import
init_db()
