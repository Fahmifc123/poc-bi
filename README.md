
# Bank Indonesia – Perception Engineering Tool  
## Proof of Concept (Streamlit)

---

## 1. Executive Summary

This Proof of Concept (POC) demonstrates a **closed-loop Perception Engineering System** designed to support Bank Indonesia in monitoring, assessing, and responding to public perception related to central bank policies.

The system simulates:

- Public perception monitoring  
- Narrative intelligence extraction  
- Risk scoring & decision support  
- AI-generated communication recommendation  
- Post-intervention impact evaluation  

This POC uses a **1-month simulated dataset** and is implemented using **Streamlit**.

---

## 2. System Architecture (Conceptual Flow)
Media Monitoring
→ Capturing Perception
→ Fact Checker (Simulated Human Layer)
→ Decision Intelligence
→ Recommendation Engine
→ DKom Communication
→ Evaluation & Impact
→ Loop Back to Monitoring

This system is designed as a **perception control loop**, rather than a passive monitoring dashboard.

---

## 3. POC Scope

### Included

- 30-day simulated dataset  
- Sentiment trend visualization  
- Topic distribution analysis  
- Emotion detection  
- Negative spike simulation  
- Narrative summary simulation  
- Risk scoring engine (rule-based)  
- AI-based communication strategy simulation  
- Before vs After intervention evaluation  

### Excluded (Production Phase)

- Real-time streaming ingestion  
- Actual document retrieval system  
- Production-grade LLM integration  
- Integration with internal BI systems  

---

## 4. Streamlit Application Structure

### Module 1 – Overview Dashboard

Displays:

- Total mentions (30 days)  
- Sentiment distribution  
- Spike alert  
- Topic trend  
- Emotion distribution  
- Time-series visualization  

---

### Module 2 – Perception Intelligence

Displays:

- Spike detail panel  
- Dominant topic  
- Narrative summary (simulated LLM output)  
- Stakeholder identification  
- Misinformation probability score  

---

### Module 3 – Risk & Decision Intelligence

Displays:

- Risk score breakdown  
- Risk classification (Low / Medium / High)  
- Decision gate simulation  

Risk score calculated from:

- Negative sentiment ratio  
- Volume & velocity  
- Influencer impact  
- Topic sensitivity  
- Misinformation score  

---

### Module 4 – Recommendation Engine

Simulated AI output:

- Suggested communication strategy  
- Recommended channel  
- Suggested tone  
- Target stakeholder  

---

### Module 5 – Evaluation & Impact

Displays:

- Pre vs Post intervention comparison  
- Sentiment shift  
- Volume change  
- Impact score  
- Intervention effectiveness status  

---

## 5. Dummy Case Scenario
Streamlit


---

## 6. Tech Stack

- Python  
- Streamlit  
- Pandas  
- Plotly  
- Rule-based risk scoring  
- Simulated LLM outputs  

---

## 7. Objective

To demonstrate how Bank Indonesia can transition from **passive monitoring** to **proactive perception engineering and strategic communication control**.