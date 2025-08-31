# ============================================================================
# app.py - Hypocrates Medical AI with MIMIC-IV Database (IMPROVED INTERFACE)
# ============================================================================

import os
import pandas as pd
import sqlite3
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import gradio as gr
import warnings

warnings.filterwarnings("ignore")

print("🏥 STARTING HYPOCRATES MEDICAL AI WITH MIMIC-IV")
print("=" * 60)

# ============================================================================
# CONFIGURATION FOR HUGGING FACE SPACES
# ============================================================================

# Detect if we are in HF Spaces
IS_HF_SPACE = os.environ.get("SPACE_ID") is not None
HF_TOKEN = os.environ.get("HF_TOKEN")  # Optional token for APIs

# Database configuration
MIMIC_DB_PATH = "hypocrates_mimic.db"
USE_REAL_MIMIC = os.path.exists(MIMIC_DB_PATH)

print(f"🌐 Running on: {'Hugging Face Spaces' if IS_HF_SPACE else 'Local'}")
print(f"🗃️ Database: {'Real MIMIC-IV' if USE_REAL_MIMIC else 'Synthetic'}")

# ============================================================================
# INTELLIGENT MEDICAL AGENT WITH MIMIC-IV
# ============================================================================


class HypocratesMedicalAgent:
    """Advanced medical agent with MIMIC-IV Emergency Department data"""

    def __init__(self):
        self.db_path = MIMIC_DB_PATH if USE_REAL_MIMIC else "hospital_synthetic.db"
        self.is_mimic = USE_REAL_MIMIC

        # Known sepsis patients (from your analysis)
        self.sepsis_patients = [10014729, 10020944, 10039708, 10019003]

        # Agent tools
        self.tools = {
            "sepsis_predictor": self.predict_sepsis_risk,
            "database_query": self.query_patient_data,
            "vital_signs_analyzer": self.analyze_vital_signs,
            "clinical_summary": self.generate_clinical_summary,
        }

        # Initialize database if needed
        if not USE_REAL_MIMIC:
            self.create_synthetic_db()

    def create_synthetic_db(self):
        """Creates synthetic database if MIMIC-IV is not available"""
        if os.path.exists("hospital_synthetic.db"):
            return

        print("🔄 Creating synthetic database (MIMIC not available)...")

        conn = sqlite3.connect("hospital_synthetic.db")
        cursor = conn.cursor()

        # Create edstays table (similar to MIMIC-IV)
        cursor.execute(
            """
            CREATE TABLE edstays (
                subject_id INTEGER,
                stay_id INTEGER,
                hadm_id INTEGER,
                gender TEXT,
                race TEXT,
                intime TIMESTAMP,
                outtime TIMESTAMP,
                disposition TEXT
            )
        """
        )

        # Create vitalsign table
        cursor.execute(
            """
            CREATE TABLE vitalsign (
                stay_id INTEGER,
                charttime TIMESTAMP,
                temperature REAL,
                heartrate INTEGER,
                resprate INTEGER,
                o2sat INTEGER,
                sbp INTEGER,
                dbp INTEGER,
                rhythm TEXT,
                pain INTEGER
            )
        """
        )

        # Create diagnosis table
        cursor.execute(
            """
            CREATE TABLE diagnosis (
                stay_id INTEGER,
                icd_code TEXT,
                icd_title TEXT,
                icd_version INTEGER
            )
        """
        )

        # Create triage table
        cursor.execute(
            """
            CREATE TABLE triage (
                stay_id INTEGER,
                temperature REAL,
                heartrate INTEGER,
                resprate INTEGER,
                o2sat INTEGER,
                sbp INTEGER,
                dbp INTEGER,
                pain INTEGER,
                acuity INTEGER,
                chiefcomplaint TEXT
            )
        """
        )

        # Insert synthetic data
        patients_data = []
        for i in range(20):
            subject_id = 10000000 + i
            stay_id = 30000000 + i
            hadm_id = 20000000 + i
            gender = random.choice(["M", "F"])
            race = random.choice(["WHITE", "BLACK", "HISPANIC", "ASIAN"])
            intime = datetime.now() - timedelta(days=random.randint(1, 30))
            outtime = intime + timedelta(hours=random.randint(4, 48))
            disposition = random.choice(["HOME", "ADMIT", "TRANSFER"])

            patients_data.append(
                (
                    subject_id,
                    stay_id,
                    hadm_id,
                    gender,
                    race,
                    intime,
                    outtime,
                    disposition,
                )
            )

        cursor.executemany(
            "INSERT INTO edstays VALUES (?, ?, ?, ?, ?, ?, ?, ?)", patients_data
        )

        # Vital signs for each patient
        for subject_id, stay_id, _, _, _, intime, _, _ in patients_data:
            for hour in range(0, 12, 2):  # Every 2 hours for 12 hours
                chart_time = intime + timedelta(hours=hour)
                temperature = round(random.uniform(97.0, 102.0), 1)
                heartrate = random.randint(60, 120)
                resprate = random.randint(12, 25)
                o2sat = random.randint(88, 100)
                sbp = random.randint(90, 160)
                dbp = random.randint(60, 100)
                pain = random.randint(0, 10)

                cursor.execute(
                    """INSERT INTO vitalsign VALUES 
                                 (?, ?, ?, ?, ?, ?, ?, ?, 'Sinus Rhythm', ?)""",
                    (
                        stay_id,
                        chart_time,
                        temperature,
                        heartrate,
                        resprate,
                        o2sat,
                        sbp,
                        dbp,
                        pain,
                    ),
                )

        # Diagnoses
        common_diagnoses = [
            ("R50.9", "Fever", 10),
            ("A41.9", "Sepsis, unspecified organism", 10),
            ("J44.1", "COPD with exacerbation", 10),
            ("I50.9", "Heart failure", 10),
        ]

        for _, stay_id, _, _, _, _, _, _ in patients_data:
            num_diag = random.randint(1, 3)
            selected_diag = random.sample(common_diagnoses, num_diag)
            for icd_code, icd_title, icd_version in selected_diag:
                cursor.execute(
                    "INSERT INTO diagnosis VALUES (?, ?, ?, ?)",
                    (stay_id, icd_code, icd_title, icd_version),
                )

        # Triage
        for _, stay_id, _, _, _, _, _, _ in patients_data:
            cursor.execute(
                """INSERT INTO triage VALUES 
                             (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    stay_id,
                    random.uniform(97, 102),
                    random.randint(60, 120),
                    random.randint(12, 25),
                    random.randint(88, 100),
                    random.randint(90, 160),
                    random.randint(60, 100),
                    random.randint(0, 10),
                    random.randint(1, 5),
                    "Fever and malaise",
                ),
            )

        conn.commit()
        conn.close()
        print(f"✅ Synthetic database created: {len(patients_data)} patients")

    def get_patient_data(self, patient_id: int) -> Dict:
        """Gets complete patient data using MIMIC-IV structure"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)

            # 1. STAY INFORMATION
            stay_query = """
            SELECT es.subject_id, es.stay_id, es.hadm_id, es.gender, es.race,
                   es.intime, es.outtime, es.disposition
            FROM edstays es
            WHERE es.subject_id = ?
            ORDER BY es.intime DESC
            LIMIT 1
            """

            stay_df = pd.read_sql_query(stay_query, conn, params=[patient_id])

            if stay_df.empty:
                return {}

            stay_info = stay_df.iloc[0].to_dict()
            stay_id = stay_info["stay_id"]

            # 2. VITAL SIGNS
            vitals_query = """
            SELECT vs.charttime, vs.temperature, vs.heartrate, vs.resprate,
                   vs.o2sat, vs.sbp, vs.dbp, vs.rhythm, vs.pain
            FROM vitalsign vs
            WHERE vs.stay_id = ?
            ORDER BY vs.charttime DESC
            LIMIT 10
            """

            vitals_df = pd.read_sql_query(vitals_query, conn, params=[stay_id])

            # 3. DIAGNOSES
            diag_query = """
            SELECT d.icd_code, d.icd_title, d.icd_version
            FROM diagnosis d
            WHERE d.stay_id = ?
            """

            diag_df = pd.read_sql_query(diag_query, conn, params=[stay_id])

            # 4. TRIAGE INFORMATION
            triage_query = """
            SELECT t.temperature, t.heartrate, t.resprate, t.o2sat,
                   t.sbp, t.dbp, t.pain, t.acuity, t.chiefcomplaint
            FROM triage t
            WHERE t.stay_id = ?
            """

            triage_df = pd.read_sql_query(triage_query, conn, params=[stay_id])

            return {
                "stay_info": stay_info,
                "vital_signs": vitals_df.to_dict("records"),
                "diagnoses": diag_df.to_dict("records"),
                "triage_info": (
                    triage_df.iloc[0].to_dict() if not triage_df.empty else {}
                ),
            }

        except Exception as e:
            print(f"Error getting patient data: {e}")
            return {}
        finally:
            if conn:
                conn.close()

    def predict_sepsis_risk(self, patient_data: Dict) -> Dict:
        """Tool: Sepsis risk prediction using proven algorithm"""
        if not patient_data:
            return {
                "risk_level": "ERROR",
                "risk_score": 0,
                "color": "⚫",
                "sirs_criteria": 0,
                "risk_factors": [
                    "Patient data not found in database"
                ],
                "recommendation": "Verify that the patient exists in MIMIC-IV",
                "patient_id": "N/A",
                "latest_vitals": {},
            }

        if not patient_data.get("vital_signs"):
            patient_id = patient_data.get("stay_info", {}).get("subject_id", "N/A")
            return {
                "risk_level": "NO DATA",
                "risk_score": 0,
                "color": "⚫",
                "sirs_criteria": 0,
                "risk_factors": [
                    "No vital signs records for this patient",
                    "Patient exists in edstays but has no data in vitalsign",
                ],
                "recommendation": "Select a patient from the 'Available Patients' list that has complete data",
                "patient_id": patient_id,
                "latest_vitals": {},
            }

        # Check data quality
        vital_signs_list = patient_data["vital_signs"]
        if len(vital_signs_list) < 3:
            patient_id = patient_data.get("stay_info", {}).get("subject_id", "N/A")
            return {
                "risk_level": "INSUFFICIENT DATA",
                "risk_score": 0,
                "color": "🟡",
                "sirs_criteria": 0,
                "risk_factors": [
                    f"Only {len(vital_signs_list)} vital signs records available",
                    "At least 3 records required for reliable analysis",
                ],
                "recommendation": "Select a patient with more vital signs records",
                "patient_id": patient_id,
                "latest_vitals": vital_signs_list[0] if vital_signs_list else {},
            }

        risk_score = 0
        risk_factors = []
        sirs_criteria = 0

        # Get latest vital signs
        latest_vitals = vital_signs_list[0]

        # 1. TEMPERATURE (SIRS criteria)
        temp = latest_vitals.get("temperature")
        if temp is not None:
            if temp > 100.4 or temp < 96.8:
                risk_score += 25
                sirs_criteria += 1
                risk_factors.append(f"Abnormal temperature: {temp}°F")

        # 2. HEART RATE (SIRS criteria)
        hr = latest_vitals.get("heartrate")
        if hr is not None and hr > 90:
            risk_score += 25
            sirs_criteria += 1
            risk_factors.append(f"Tachycardia: {hr} bpm > 90")

        # 3. RESPIRATORY RATE (SIRS criteria)
        rr = latest_vitals.get("resprate")
        if rr is not None and rr > 20:
            risk_score += 25
            sirs_criteria += 1
            risk_factors.append(f"Tachypnea: {rr} rpm > 20")

        # 4. OXYGEN SATURATION
        o2sat = latest_vitals.get("o2sat")
        if o2sat is not None and o2sat < 92:
            risk_score += 30
            risk_factors.append(f"Hypoxemia: SpO2 {o2sat}% < 92")

        # 5. BLOOD PRESSURE (qSOFA criteria)
        sbp = latest_vitals.get("sbp")
        if sbp is not None and sbp < 100:
            risk_score += 30
            risk_factors.append(f"Hypotension: SBP {sbp} mmHg < 100")

        # 6. DIAGNOSIS OF SEPSIS
        diagnoses = patient_data.get("diagnoses", [])
        sepsis_dx = any(
            "sepsis" in str(d.get("icd_title", "")).lower() for d in diagnoses
        )
        if sepsis_dx:
            risk_score += 50
            risk_factors.append("🚨 CONFIRMED SEPSIS DIAGNOSIS")

        # DETERMINE RISK LEVEL
        if risk_score >= 70:
            risk_level = "CRITICAL"
            color = "🔴"
            recommendation = "SEPSIS ALERT: Immediate sepsis protocol"
        elif risk_score >= 45:
            risk_level = "HIGH"
            color = "🟠"
            recommendation = "High sepsis risk: Continuous monitoring"
        elif risk_score >= 25:
            risk_level = "MODERATE"
            color = "🟡"
            recommendation = "Moderate risk: Close surveillance"
        else:
            risk_level = "LOW"
            color = "🟢"
            recommendation = "Low risk: Routine monitoring"

        return {
            "risk_level": risk_level,
            "risk_score": min(risk_score, 100),
            "color": color,
            "sirs_criteria": sirs_criteria,
            "risk_factors": risk_factors,
            "recommendation": recommendation,
            "patient_id": patient_data["stay_info"]["subject_id"],
            "latest_vitals": latest_vitals,
        }

    def analyze_vital_signs(self, patient_data: Dict) -> str:
        """Tool: Detailed analysis using MIMIC-IV structure"""
        if not patient_data:
            return "Patient data not found."

        vital_signs = patient_data.get("vital_signs", [])
        if not vital_signs:
            return "No vital signs found for this patient."

        patient_id = patient_data["stay_info"].get("subject_id", "Unknown")

        result = f"## ❤️ VITAL SIGNS ANALYSIS - Patient {patient_id}\n\n"
        result += f"**Source:** {'MIMIC-IV Emergency Department' if self.is_mimic else 'Synthetic Database'}\n\n"

        # Analyze most recent vital signs
        latest_vitals = vital_signs[0]

        vital_analyses = []

        # Temperature
        temp = latest_vitals.get("temperature")
        if temp is not None:
            if temp > 100.4:
                status, interpretation = "🔴", "FEVER"
            elif temp < 96.8:
                status, interpretation = "🔵", "HYPOTHERMIA"
            else:
                status, interpretation = "✅", "Normal"
            vital_analyses.append((f"Temperature: {temp}°F", status, interpretation))

        # Heart Rate
        hr = latest_vitals.get("heartrate")
        if hr is not None:
            if hr > 100:
                status, interpretation = "🔴", "TACHYCARDIA"
            elif hr < 60:
                status, interpretation = "🔵", "BRADYCARDIA"
            else:
                status, interpretation = "✅", "Normal"
            vital_analyses.append(
                (f"Heart Rate: {hr} bpm", status, interpretation)
            )

        # Respiratory Rate
        rr = latest_vitals.get("resprate")
        if rr is not None:
            if rr > 20:
                status, interpretation = "🔴", "TACHYPNEA"
            elif rr < 12:
                status, interpretation = "🔵", "BRADYPNEA"
            else:
                status, interpretation = "✅", "Normal"
            vital_analyses.append(
                (f"Respiratory Rate: {rr} rpm", status, interpretation)
            )

        # Oxygen Saturation
        o2sat = latest_vitals.get("o2sat")
        if o2sat is not None:
            if o2sat < 92:
                status, interpretation = "🔴", "SEVERE HYPOXEMIA"
            elif o2sat < 95:
                status, interpretation = "🟠", "MILD HYPOXEMIA"
            else:
                status, interpretation = "✅", "Normal"
            vital_analyses.append((f"O2 Saturation: {o2sat}%", status, interpretation))

        # Blood Pressure
        sbp = latest_vitals.get("sbp")
        dbp = latest_vitals.get("dbp")
        if sbp is not None:
            if sbp < 100:
                status, interpretation = "🔴", "HYPOTENSION"
            elif sbp > 140:
                status, interpretation = "🟠", "HYPERTENSION"
            else:
                status, interpretation = "✅", "Normal"

            bp_text = f"Blood Pressure: {sbp}"
            if dbp is not None:
                bp_text += f"/{dbp}"
            bp_text += " mmHg"
            vital_analyses.append((bp_text, status, interpretation))

        # Show analysis
        for vital_text, status, interpretation in vital_analyses:
            result += f"{status} **{vital_text}** - *{interpretation}*\n"

        return result

    def query_patient_data(self, patient_id: int, query_type: str = "basic") -> str:
        """Tool: Specific data query using MIMIC-IV structure"""
        patient_data = self.get_patient_data(patient_id)

        if not patient_data:
            return f"No data found for patient {patient_id} in {'MIMIC-IV ED' if self.is_mimic else 'database'}."

        stay_info = patient_data["stay_info"]

        result = f"## 👤 PATIENT DATA {patient_id}\n\n"
        result += f"**Source:** {'MIMIC-IV Emergency Department' if self.is_mimic else 'Synthetic Data'}\n\n"

        result += f"**Patient ID:** {stay_info.get('subject_id', 'N/A')}\n"
        result += f"**Stay ID:** {stay_info.get('stay_id', 'N/A')}\n"
        result += f"**Gender:** {stay_info.get('gender', 'N/A')}\n"
        result += f"**Race:** {stay_info.get('race', 'N/A')}\n"
        result += f"**Admission Time:** {stay_info.get('intime', 'N/A')}\n"
        result += f"**Discharge Time:** {stay_info.get('outtime', 'N/A')}\n"
        result += f"**Disposition:** {stay_info.get('disposition', 'N/A')}\n\n"

        # Diagnoses
        diagnoses = patient_data.get("diagnoses", [])
        if diagnoses:
            result += f"**Diagnoses ({len(diagnoses)}):**\n"
            for diag in diagnoses[:5]:  # Show first 5
                title = diag.get("icd_title", "N/A")
                code = diag.get("icd_code", "N/A")
                version = diag.get("icd_version", "N/A")
                # Highlight sepsis
                if "sepsis" in title.lower():
                    result += f"- 🔴 **{title}** ({code}, ICD-{version})\n"
                else:
                    result += f"- {title} ({code}, ICD-{version})\n"

        return result

    def generate_clinical_summary(self, patient_data: Dict) -> str:
        """Tool: Generate clinical summary using MIMIC-IV structure"""
        if not patient_data:
            return "Cannot generate summary - insufficient data."

        # Get sepsis analysis
        sepsis_analysis = self.predict_sepsis_risk(patient_data)

        stay_info = patient_data["stay_info"]
        patient_id = stay_info.get("subject_id", "Unknown")

        result = f"## 📋 COMPLETE CLINICAL SUMMARY - Patient {patient_id}\n\n"
        result += f"**Source:** {'MIMIC-IV Emergency Department' if self.is_mimic else 'Synthetic Data'}\n\n"

        # Sepsis assessment
        result += (
            f"### {sepsis_analysis.get('color', '⚪')} Sepsis Risk Assessment\n"
        )
        result += f"- **Risk Level:** {sepsis_analysis.get('risk_level', 'Unknown')}\n"
        result += f"- **Score:** {sepsis_analysis.get('risk_score', 0)}/100\n"
        result += (
            f"- **SIRS Criteria:** {sepsis_analysis.get('sirs_criteria', 0)}/3\n\n"
        )

        # Risk factors
        if sepsis_analysis.get("risk_factors"):
            result += "### 🔍 Identified Risk Factors\n"
            for factor in sepsis_analysis["risk_factors"]:
                result += f"- {factor}\n"
            result += "\n"

        # Clinical recommendation
        result += "### 💡 Clinical Recommendation\n"
        result += f"{sepsis_analysis.get('recommendation', 'No recommendations available')}\n\n"

        return result

    def get_available_patients(self) -> List[int]:
        """Gets list of patients that DO have complete data for analysis"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)

            # Only patients with COMPLETE data for sepsis analysis
            query = """
            SELECT DISTINCT e.subject_id, 
                   COUNT(v.stay_id) as vital_count,
                   COUNT(d.stay_id) as diag_count
            FROM edstays e
            LEFT JOIN vitalsign v ON e.stay_id = v.stay_id
            LEFT JOIN diagnosis d ON e.stay_id = d.stay_id
            GROUP BY e.subject_id
            HAVING vital_count >= 5 AND diag_count >= 1
            ORDER BY e.subject_id
            LIMIT 50
            """

            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            patients = [row[0] for row in results]

            print(f"Patients with complete data found: {len(patients)}")

            if self.is_mimic:
                # Prioritize known sepsis patients that DO have data
                sepsis_patients_with_data = [
                    p for p in self.sepsis_patients if p in patients
                ]
                other_patients = [p for p in patients if p not in self.sepsis_patients]

                # Check which sepsis patients have data
                missing_sepsis = [p for p in self.sepsis_patients if p not in patients]
                if missing_sepsis:
                    print(
                        f"⚠️ Sepsis patients without sufficient data: {missing_sepsis}"
                    )

                return (
                    sepsis_patients_with_data + other_patients[:30]
                )  # Limit to 30 total
            else:
                return patients

        except Exception as e:
            print(f"Error getting patients with complete data: {e}")
            # Fallback: search for any patient that has at least vital signs
            try:
                fallback_query = """
                SELECT DISTINCT e.subject_id
                FROM edstays e
                JOIN vitalsign v ON e.stay_id = v.stay_id
                ORDER BY e.subject_id
                LIMIT 20
                """
                cursor.execute(fallback_query)
                patients = [row[0] for row in cursor.fetchall()]
                print(f"Using fallback: {len(patients)} patients with vital signs")
                return patients
            except:
                return (
                    self.sepsis_patients
                    if self.is_mimic
                    else list(range(10000000, 10000020))
                )
        finally:
            if conn:
                conn.close()

    def run(self, query: str) -> str:
        """Main agent engine - decides which tool to use"""
        try:
            query_lower = query.lower()

            # Extract patient_id from query
            patient_id = None
            words = query.split()
            for i, word in enumerate(words):
                if word.lower() in ["patient", "paciente", "subject"] and i + 1 < len(
                    words
                ):
                    try:
                        patient_id = int(words[i + 1].replace(":", ""))
                        break
                    except ValueError:
                        continue

            if not patient_id:
                # Search for ID anywhere in the query
                import re

                numbers = re.findall(r"\d{8,}", query)  # Search for numbers with 8+ digits
                if numbers:
                    patient_id = int(numbers[0])
                else:
                    available = self.get_available_patients()
                    patient_list = ", ".join(map(str, available[:10]))
                    return f"❌ **Error:** Please specify a valid patient ID.\n\n📋 **Available patients:** {patient_list}"

            # Get patient data
            patient_data = self.get_patient_data(patient_id)
            if not patient_data:
                return f"❌ **Error:** Patient {patient_id} not found in {'MIMIC-IV' if self.is_mimic else 'database'}."

            # Decide which tool to use based on query
            if any(
                word in query_lower
                for word in ["sepsis", "risk", "riesgo", "infection"]
            ):
                # Use sepsis prediction tool
                analysis = self.predict_sepsis_risk(patient_data)

                result = f"""
## {analysis['color']} SEPSIS RISK ASSESSMENT
**Patient ID:** {analysis['patient_id']}  
**Data Source:** {'MIMIC-IV (Real)' if self.is_mimic else 'Synthetic'}  
**Risk Level:** {analysis['risk_level']}  
**Score:** {analysis['risk_score']}/100  
**SIRS Criteria:** {analysis['sirs_criteria']}/3  
### 🔍 Identified Risk Factors:
{chr(10).join([f"• {factor}" for factor in analysis['risk_factors']]) if analysis['risk_factors'] else "• No significant risk factors detected"}
### 💡 Clinical Recommendation:
{analysis['recommendation']}
### 📊 Current Vital Signs:
"""
                for key, value in analysis["latest_vitals"].items():
                    if value is not None and key != "charttime":
                        unit_map = {
                            "temperature": "°F",
                            "heartrate": "bpm",
                            "resprate": "rpm",
                            "o2sat": "%",
                            "sbp": "mmHg",
                            "dbp": "mmHg",
                        }
                        unit = unit_map.get(key, "")
                        result += f"• **{key.title()}:** {value} {unit}\n"

                return result

            elif any(word in query_lower for word in ["vital", "signos", "signs"]):
                # Use vital signs analysis tool
                return self.analyze_vital_signs(patient_data)

            elif any(
                word in query_lower
                for word in ["summary", "resumen", "complete", "completo"]
            ):
                # Use clinical summary tool
                return self.generate_clinical_summary(patient_data)

            elif any(
                word in query_lower for word in ["data", "datos", "info", "información"]
            ):
                # Use data query tool
                return self.query_patient_data(patient_id)

            else:
                # General analysis - use multiple tools
                basic_info = self.query_patient_data(patient_id)
                sepsis_risk = self.predict_sepsis_risk(patient_data)

                return f"""
{basic_info}
### 🚨 Sepsis Risk Assessment
**Level:** {sepsis_risk['risk_level']} ({sepsis_risk['risk_score']}/100 points)  
**Recommendation:** {sepsis_risk['recommendation']}
💡 *For more detailed analysis, ask specifically about sepsis risk, vital signs or clinical summary.*
"""

        except Exception as e:
            return f"❌ **Analysis Error:** {str(e)}"


# ============================================================================
# GRADIO INTERFACE FUNCTIONS
# ============================================================================


def query_agent(patient_id, question):
    """Executes query on medical agent"""
    try:
        if not question or not question.strip():
            return "❌ **Error:** Please enter a question."

        # Format query including patient_id if provided
        if patient_id is not None:
            query_text = f"For patient {int(patient_id)}: {question}"
        else:
            query_text = question

        result = medical_agent.run(query_text)
        return result

    except Exception as e:
        return f"❌ **Error:** {str(e)}"


def get_patient_info(patient_id):
    """Gets basic patient information using MIMIC-IV structure"""
    try:
        if patient_id is None:
            return "⚠️ **Select a patient ID**"

        patient_data = medical_agent.get_patient_data(int(patient_id))

        if not patient_data:
            available = medical_agent.get_available_patients()
            if available:
                patient_list = ", ".join(map(str, available[:10]))
                sepsis_note = ""
                if USE_REAL_MIMIC:
                    sepsis_patients_str = ", ".join(
                        map(str, medical_agent.sepsis_patients)
                    )
                    sepsis_note = f"\n\n🔴 **Patients with confirmed sepsis:** {sepsis_patients_str}"
                return f"❌ **Patient {patient_id} not found**\n\n🔍 **Available:** {patient_list}{sepsis_note}"
            else:
                return f"❌ **Patient {patient_id} not found**"

        stay_info = patient_data["stay_info"]

        info = f"""
## 👤 PATIENT INFORMATION
**ID:** {stay_info.get('subject_id', 'N/A')}  
**Source:** {'MIMIC-IV Emergency Department' if USE_REAL_MIMIC else 'Synthetic Data'}  
**Gender:** {stay_info.get('gender', 'N/A')} | **Race:** {stay_info.get('race', 'N/A')}  
**Emergency Department Stay:**  
**Stay ID:** {stay_info.get('stay_id', 'N/A')}  
**Admission:** {stay_info.get('intime', 'N/A')}  
**Discharge:** {stay_info.get('outtime', 'N/A')}  
**Disposition:** {stay_info.get('disposition', 'N/A')}  
**Available Data:**
- Vital Signs: {len(patient_data.get('vital_signs', []))} records
- Diagnoses: {len(patient_data.get('diagnoses', []))} diagnoses
- Triage: {'Yes' if patient_data.get('triage_info') else 'No'}
"""

        # Highlight if patient has confirmed sepsis
        if USE_REAL_MIMIC and int(patient_id) in medical_agent.sepsis_patients:
            info += "🔴 **PATIENT WITH CONFIRMED SEPSIS IN MIMIC-IV**\n\n"

        info += (
            "---\n✅ **Data loaded from MIMIC-IV ED**"
            if USE_REAL_MIMIC
            else "✅ **Synthetic data loaded**"
        )

        return info

    except Exception as e:
        return f"❌ **Error:** {str(e)}"


def get_available_patients_display():
    """Lists available patients from MIMIC-IV database"""
    try:
        patients = medical_agent.get_available_patients()

        result = f"## 👥 AVAILABLE PATIENTS\n\n"
        result += f"**Source:** {'MIMIC-IV Emergency Department' if USE_REAL_MIMIC else 'Synthetic Database'}\n"
        result += f"**Total Available:** {len(patients)} patients\n\n"

        if USE_REAL_MIMIC:
            # Show sepsis patients first
            result += "### 🔴 Patients with Confirmed Sepsis\n"
            sepsis_available = [
                p for p in patients if p in medical_agent.sepsis_patients
            ]
            for patient_id in sepsis_available[:4]:  # Show first 4
                patient_data = medical_agent.get_patient_data(patient_id)
                if patient_data:
                    stay_info = patient_data["stay_info"]
                    result += f"**{patient_id}** - {stay_info.get('gender', '?')}, {stay_info.get('race', 'Unknown')}\n"
                else:
                    result += f"**{patient_id}** - Data not available\n"

            result += "\n### ⚪ Other Available Patients\n"
            other_patients = [
                p for p in patients if p not in medical_agent.sepsis_patients
            ]
            for i, patient_id in enumerate(other_patients[:10]):  # Show first 10
                patient_data = medical_agent.get_patient_data(patient_id)
                if patient_data:
                    stay_info = patient_data["stay_info"]
                    result += f"**{patient_id}** - {stay_info.get('gender', '?')}, {stay_info.get('race', 'Unknown')}\n"
                else:
                    result += f"**{patient_id}** - Data not available\n"
        else:
            # Synthetic data
            for i, patient_id in enumerate(patients[:15]):  # Show first 15
                if i < 10:
                    patient_data = medical_agent.get_patient_data(patient_id)
                    if patient_data:
                        stay_info = patient_data["stay_info"]
                        result += f"**{patient_id}** - {stay_info.get('gender', '?')}, {stay_info.get('race', 'Unknown')}\n"

        if len(patients) > 15:
            result += f"\n... and {len(patients) - 15} more patients available."

        return result

    except Exception as e:
        return f"❌ **Error:** {str(e)}"


# ============================================================================
# SYSTEM INITIALIZATION
# ============================================================================

# Create medical agent
medical_agent = HypocratesMedicalAgent()

# Get initial patient list
available_patients = medical_agent.get_available_patients()

if USE_REAL_MIMIC and available_patients:
    # Use real MIMIC-IV ranges
    min_patient = min(available_patients)
    max_patient = max(available_patients)
    default_patient = (
        medical_agent.sepsis_patients[0]
        if medical_agent.sepsis_patients
        else min_patient
    )
else:
    # Fallback for synthetic data
    min_patient = 10000000
    max_patient = 10000020
    default_patient = 10000001

print(f"✅ Medical agent initialized")
print(f"📊 Available patients: {len(available_patients)}")
print(f"🔢 ID range: {min_patient} - {max_patient}")
if USE_REAL_MIMIC:
    print(f"🔴 Sepsis patients: {medical_agent.sepsis_patients}")
else:
    print("🔴 Synthetic mode activated")

# ============================================================================
# IMPROVED GRADIO INTERFACE (NEW DESIGN)
# ============================================================================

custom_css = """
.gradio-container {  max-width: 1400px !important; 
    margin: 0 auto !important; }
.medical-header {
    background: linear-gradient(135deg, #1e40af 0%, #7c3aed 100%);
    padding: 25px; border-radius: 15px; color: white; text-align: center; margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.mimic-badge {
    background: #059669; color: white; padding: 4px 12px; 
    border-radius: 20px; font-size: 12px; font-weight: bold;
}
.synthetic-badge {
    background: #dc2626; color: white; padding: 4px 12px; 
    border-radius: 20px; font-size: 12px; font-weight: bold;
}
.risk-critical { background: #fee2e2; border-left: 4px solid #dc2626; padding: 10px; border-radius: 5px; }
.risk-high { background: #ffedd5; border-left: 4px solid #ea580c; padding: 10px; border-radius: 5px; }
.risk-moderate { background: #fef3c7; border-left: 4px solid #d97706; padding: 10px; border-radius: 5px; }
.risk-low { background: #dcfce7; border-left: 4px solid #16a34a; padding: 10px; border-radius: 5px; }
.vital-card { background: #f8fafc; padding: 15px; border-radius: 10px; margin: 10px 0; border: 1px solid #e2e8f0; }
.tab-button { font-weight: bold; }
"""

# Create interface
with gr.Blocks(
    title="Hypocrates Medical AI - MIMIC-IV Edition",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),  # type: ignore
    css=custom_css,
) as demo:

    # Header
    badge_class = "mimic-badge" if USE_REAL_MIMIC else "synthetic-badge"
    badge_text = "MIMIC-IV Real Data" if USE_REAL_MIMIC else "Synthetic Data"

    gr.HTML(
        f"""
    <div class="medical-header">
        <h1 style="font-size: 2.5rem; margin-bottom: 10px;">🏥 Hypocrates Medical AI</h1>
        <p style="font-size: 1.2rem; margin-bottom: 15px;">Advanced Medical Analysis System with Intelligent Agent</p>
        <span class="{badge_class}">{badge_text}</span>
        <p style="margin-top: 15px; font-size: 0.9rem;">Sepsis analysis based on MIMIC-IV Emergency Department data</p>
    </div>
    """
    )

    with gr.Row():
        # Left panel - Patient management
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("## 📋 Patient Management")

            with gr.Row():
                patient_id = gr.Dropdown(
                    label="Select Patient",
                    choices=[str(p) for p in available_patients],
                    value=str(default_patient),
                    interactive=True,
                    info="Select a patient to analyze",
                )
                refresh_btn = gr.Button("🔄", variant="secondary", size="sm")

            with gr.Row():
                load_info_btn = gr.Button(
                    "👤 Load Info", variant="primary", size="sm"
                )
                list_patients_btn = gr.Button(
                    "👥 View List", variant="secondary", size="sm"
                )

            patient_info = gr.Markdown(
                value="*Select a patient to get started...*",
                label="Patient Information",
            )

        # Central panel - Analysis and queries
        with gr.Column(scale=2, min_width=500):
            gr.Markdown("## 🤖 Intelligent Medical Agent")

            with gr.Tab("💬 Medical Query", elem_classes="tab-button"):
                question = gr.Textbox(
                    label="Medical Query",
                    placeholder="What is the sepsis risk for this patient? Please provide detailed analysis.",
                    lines=3,
                    info="The agent will automatically decide which tools to use",
                )

                submit_btn = gr.Button(
                    "🚀 Query Agent", variant="primary", size="lg"
                )

                # Improved examples
                gr.Markdown("### 💡 Query Examples")
                gr.Examples(
                    examples=[
                        [
                            "What is the sepsis risk for this patient? Give me detailed analysis."
                        ],
                        [
                            "Analyze all vital signs and give clinical interpretation."
                        ],
                        ["Generate a complete clinical summary for this patient."],
                        ["What data do you have available for this patient?"],
                        ["Evaluate risk factors and give clinical recommendations."],
                        ["Are there any alarm signs in the vital signs?"],
                    ],
                    inputs=[question],
                )

            with gr.Tab("📊 Quick Analysis", elem_classes="tab-button"):
                gr.Markdown("### Automatic Data Analysis")

                with gr.Row():
                    sepsis_btn = gr.Button(
                        "🔴 Evaluate Sepsis Risk", variant="primary"
                    )
                    vitals_btn = gr.Button(
                        "❤️ Analyze Vital Signs", variant="secondary"
                    )
                    summary_btn = gr.Button("📋 Clinical Summary", variant="secondary")

                gr.Markdown("---")
                gr.Markdown("**Analysis Results:**")
                quick_output = gr.Markdown(
                    value="*Select an analysis type to get started...*"
                )

            output = gr.Markdown(
                label="🧠 Medical Agent Response",
                value=f"""
### 🎯 Medical Agent Ready
**Data Source:** {'MIMIC-IV Emergency Department (Real Data)' if USE_REAL_MIMIC else 'Synthetic Database'}  
**Available Patients:** {len(available_patients)}  
**Agent Capabilities:**
🔍 **Available Tools:**
- Sepsis Risk Predictor (SIRS Criteria)
- Vital Signs Analyzer  
- Database Query Tool
- Clinical Summary Generator
The agent will automatically select appropriate tools based on your query.
💡 **Instructions:**
1. Select a patient (ID: {min_patient}-{max_patient})
2. Make your specific medical query
3. The agent will analyze and respond using necessary tools
                """,
            )

        # Right panel - Monitoring dashboard
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("## 📈 Monitoring Dashboard")

            # Patient status card
            patient_status = gr.Markdown(
                value="**Patient Status:** *Not selected*",
                label="Current Status",
            )

            # Sepsis risk card
            sepsis_risk = gr.Markdown(
                value="**Sepsis Risk:** *Not evaluated*",
                label="Sepsis Assessment",
            )

            # Vital signs card
            vitals_card = gr.Markdown(
                value="**Vital Signs:** *Not available*",
                label="Recent Vital Signs",
            )

            # Quick statistics
            stats_card = gr.Markdown(
                value=f"""
**System Statistics:**
- Patients in DB: {len(available_patients)}
- Data: {'MIMIC-IV Real' if USE_REAL_MIMIC else 'Synthetic'}
- Sepsis patients: {len(medical_agent.sepsis_patients) if USE_REAL_MIMIC else 'N/A'}
                """,
                label="Statistics",
            )

    # Functions for quick analysis
    def analyze_sepsis_risk(patient_id):
        if not patient_id:
            return "⚠️ **Select a patient first**"
        patient_data = medical_agent.get_patient_data(int(patient_id))
        if not patient_data:
            return "❌ **Patient not found**"
        analysis = medical_agent.predict_sepsis_risk(patient_data)

        risk_class = ""
        if analysis["risk_level"] == "CRITICAL":
            risk_class = "risk-critical"
        elif analysis["risk_level"] == "HIGH":
            risk_class = "risk-high"
        elif analysis["risk_level"] == "MODERATE":
            risk_class = "risk-moderate"
        else:
            risk_class = "risk-low"

        result = f"""
<div class="{risk_class}">
{analysis['color']} **SEPSIS RISK ASSESSMENT**
**Patient:** {analysis['patient_id']}  
**Risk Level:** {analysis['risk_level']}  
**Score:** {analysis['risk_score']}/100  
**SIRS Criteria:** {analysis['sirs_criteria']}/3  
**Recommendation:** {analysis['recommendation']}
</div>
**Identified Risk Factors:**
"""
        for factor in analysis["risk_factors"]:
            result += f"• {factor}\n"

        return result

    def analyze_vitals(patient_id):
        if not patient_id:
            return "⚠️ **Select a patient first**", None
        patient_data = medical_agent.get_patient_data(int(patient_id))
        if not patient_data:
            return "❌ **Patient not found**", None
        result = medical_agent.analyze_vital_signs(patient_data)
        return result

    def generate_summary(patient_id):
        if not patient_id:
            return "⚠️ **Select a patient first**", None
        patient_data = medical_agent.get_patient_data(int(patient_id))
        if not patient_data:
            return "❌ **Patient not found**", None
        result = medical_agent.generate_clinical_summary(patient_data)
        return result

    def update_dashboard(patient_id):
        if not patient_id:
            return (
                "**Patient Status:** *Not selected*",
                "**Sepsis Risk:** *Not evaluated*",
                "**Vital Signs:** *Not available*",
            )

        patient_data = medical_agent.get_patient_data(int(patient_id))
        if not patient_data:
            return (
                "❌ **Patient not found**",
                "❌ **Data not available**",
                "❌ **Data not available**",
            )

        # Patient status
        stay_info = patient_data["stay_info"]
        patient_status_text = f"""
**Patient Status:** {stay_info.get('subject_id', 'N/A')}
**Gender:** {stay_info.get('gender', 'N/A')} | **Race:** {stay_info.get('race', 'N/A')}
**Disposition:** {stay_info.get('disposition', 'N/A')}
"""

        # Sepsis risk
        sepsis_analysis = medical_agent.predict_sepsis_risk(patient_data)
        sepsis_text = f"""
**Sepsis Risk:** {sepsis_analysis['risk_level']} ({sepsis_analysis['risk_score']}/100)
**SIRS Criteria:** {sepsis_analysis['sirs_criteria']}/3
"""

        # Vital signs
        vitals_text = "**Recent Vital Signs:**\n"
        if patient_data.get("vital_signs"):
            latest_vitals = patient_data["vital_signs"][0]
            for key, value in latest_vitals.items():
                if value is not None and key != "charttime":
                    unit_map = {
                        "temperature": "°F",
                        "heartrate": "bpm",
                        "resprate": "rpm",
                        "o2sat": "%",
                        "sbp": "mmHg",
                        "dbp": "mmHg",
                    }
                    unit = unit_map.get(key, "")
                    vitals_text += f"• {key.title()}: {value} {unit}\n"
        else:
            vitals_text += "*Not available*"

        return patient_status_text, sepsis_text, vitals_text

    # Event handlers
    load_info_btn.click(
        fn=get_patient_info, inputs=[patient_id], outputs=[patient_info]
    ).then(
        fn=update_dashboard,
        inputs=[patient_id],
        outputs=[patient_status, sepsis_risk, vitals_card],
    )

    list_patients_btn.click(
        fn=get_available_patients_display, inputs=[], outputs=[patient_info]
    )

    submit_btn.click(
        fn=query_agent, inputs=[patient_id, question], outputs=[output]
    ).then(
        fn=update_dashboard,
        inputs=[patient_id],
        outputs=[patient_status, sepsis_risk, vitals_card],
    )

    patient_id.change(
        fn=get_patient_info, inputs=[patient_id], outputs=[patient_info]
    ).then(
        fn=update_dashboard,
        inputs=[patient_id],
        outputs=[patient_status, sepsis_risk, vitals_card],
    )

    refresh_btn.click(
        fn=lambda: gr.Dropdown(
            choices=[str(p) for p in medical_agent.get_available_patients()],
            value=str(default_patient),
        ),
        outputs=[patient_id],
    )

    # Handlers for quick analysis
    sepsis_btn.click(
        fn=analyze_sepsis_risk, inputs=[patient_id], outputs=[quick_output]
    ).then(
        fn=update_dashboard,
        inputs=[patient_id],
        outputs=[patient_status, sepsis_risk, vitals_card],
    )

    vitals_btn.click(
        fn=analyze_vitals, inputs=[patient_id], outputs=[quick_output]
    ).then(
        fn=update_dashboard,
        inputs=[patient_id],
        outputs=[patient_status, sepsis_risk, vitals_card],
    )

    summary_btn.click(
        fn=generate_summary, inputs=[patient_id], outputs=[quick_output]
    ).then(
        fn=update_dashboard,
        inputs=[patient_id],
        outputs=[patient_status, sepsis_risk, vitals_card],
    )

# ============================================================================
# LAUNCH APPLICATION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🏥 HYPOCRATES MEDICAL AI - STARTING SERVER")
    print("=" * 60)
    print(f"🗃️ Database: {'MIMIC-IV (Real)' if USE_REAL_MIMIC else 'Synthetic'}")
    print(f"👥 Patients: {len(available_patients)}")
    print(f"🧠 Agent: Intelligent with 4 medical tools")
    print(f"🌐 {'Hugging Face Spaces' if IS_HF_SPACE else 'Local Server'}")
    print("=" * 60)

    # For HF Spaces, use default configuration
    if IS_HF_SPACE:
        demo.launch()
    else:
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
