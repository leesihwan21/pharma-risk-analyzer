# Pharma Risk Analyzer

> AI-powered Drug Adverse Event Risk Analysis & Clinical Decision Support System

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-green)](https://flask.palletsprojects.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)](https://ultralytics.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.0-red)](https://xgboost.ai)
[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://pharma-risk-analyzer-production.up.railway.app)

## Live Demo
**https://pharma-risk-analyzer-production.up.railway.app**

## Features
- FDA FAERS 2024 Q1~2025 Q1 data analysis (~480K rows)
- AI Risk Prediction (XGBoost)
- SHAP Explainable AI visualization
- YOLOv8 pill image detection
- Drug Lookup (Korea MFDS + OpenFDA)
- Drug-Drug Interaction Checker
- Dosage Calculator (CrCl/Cockcroft-Gault, pediatric, BSA)
- PRR Signal Detection (Evans criteria)
- AE Manager with ICH E2B(R3) XML export
- Korean/English i18n

## Tech Stack
- Backend: Python, Flask, SQLAlchemy
- ML/AI: XGBoost, SHAP, YOLOv8, scikit-learn
- Frontend: HTML, CSS, JavaScript, Plotly
- Deployment: Railway

## GitHub
https://github.com/leesihwan21/pharma-risk-analyzer
