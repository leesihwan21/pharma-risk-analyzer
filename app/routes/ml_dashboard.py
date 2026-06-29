import os
import json
from flask import Blueprint, jsonify, render_template, current_app

ml_dashboard = Blueprint('ml_dashboard', __name__)

# ── 하드코딩 경로 제거 → current_app.config 사용 ──────────────

def get_mlflow_runs():
    try:
        import sqlite3
        db_path = current_app.config['MLFLOW_DB_PATH']
        if not os.path.exists(db_path):
            return []

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        runs = conn.execute("""
            SELECT r.run_uuid, r.name, r.status, r.start_time, r.end_time,
                   e.name as experiment_name
            FROM runs r
            JOIN experiments e ON r.experiment_id = e.experiment_id
            WHERE r.lifecycle_stage = 'active'
            ORDER BY r.start_time DESC
            LIMIT 20
        """).fetchall()

        result = []
        for run in runs:
            run_id  = run['run_uuid']
            metrics = conn.execute("""
                SELECT key, value FROM latest_metrics WHERE run_uuid = ?
            """, (run_id,)).fetchall()
            metric_map = {m['key']: round(m['value'], 4) for m in metrics}

            params = conn.execute("""
                SELECT key, value FROM params WHERE run_uuid = ?
            """, (run_id,)).fetchall()
            param_map = {p['key']: p['value'] for p in params}

            start_ms = run['start_time']
            end_ms   = run['end_time']
            duration = round((end_ms - start_ms) / 1000, 1) if end_ms and start_ms else None

            result.append({
                'run_id':         run_id[:8],
                'run_id_full':    run_id,
                'name':           run['name'] or 'unnamed',
                'status':         run['status'],
                'experiment':     run['experiment_name'],
                'start_time':     start_ms,
                'duration_sec':   duration,
                'accuracy':       metric_map.get('accuracy'),
                'f1_risk':        metric_map.get('f1_risk'),
                'recall_risk':    metric_map.get('recall_risk'),
                'precision_risk': metric_map.get('precision_risk'),
                'cv_best_f1':     metric_map.get('cv_best_f1'),
                'n_estimators':   param_map.get('n_estimators'),
                'max_depth':      param_map.get('max_depth'),
                'learning_rate':  param_map.get('learning_rate'),
                'quarter':        param_map.get('quarter', '-'),
                'train_size':     param_map.get('train_size'),
            })

        conn.close()
        return result

    except Exception as e:
        return [{'error': str(e)}]


def get_pipeline_log():
    try:
        log_path = current_app.config['PIPELINE_LOG_PATH']
        if not os.path.exists(log_path):
            return []
        with open(log_path) as f:
            return json.load(f)
    except:
        return []


@ml_dashboard.route('/ml_dashboard')
def ml_dashboard_page():
    return render_template('ml_dashboard.html')


@ml_dashboard.route('/api/ml_dashboard/runs')
def api_runs():
    runs = get_mlflow_runs()
    log  = get_pipeline_log()
    return jsonify({
        'runs':         runs,
        'pipeline_log': log,
        'db_exists':    os.path.exists(current_app.config['MLFLOW_DB_PATH'])
    })