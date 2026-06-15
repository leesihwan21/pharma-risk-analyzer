import os
import json
from flask import Blueprint, jsonify, render_template

ml_dashboard = Blueprint('ml_dashboard', __name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH  = os.path.join(BASE_DIR, 'mlflow.db')
LOG_PATH = os.path.join(BASE_DIR, 'ml', 'pipeline_log.json')


def get_mlflow_runs():
    """mlflow.db에서 실험 run 목록 조회"""
    try:
        import sqlite3
        if not os.path.exists(DB_PATH):
            return []

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        # runs 조회
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
            run_id = run['run_uuid']

            # 메트릭 조회
            metrics = conn.execute("""
                SELECT key, value FROM latest_metrics WHERE run_uuid = ?
            """, (run_id,)).fetchall()
            metric_map = {m['key']: round(m['value'], 4) for m in metrics}

            # 파라미터 조회
            params = conn.execute("""
                SELECT key, value FROM params WHERE run_uuid = ?
            """, (run_id,)).fetchall()
            param_map = {p['key']: p['value'] for p in params}

            start_ms = run['start_time']
            end_ms   = run['end_time']
            duration = round((end_ms - start_ms) / 1000, 1) if end_ms and start_ms else None

            result.append({
                'run_id':          run_id[:8],
                'run_id_full':     run_id,
                'name':            run['name'] or 'unnamed',
                'status':          run['status'],
                'experiment':      run['experiment_name'],
                'start_time':      start_ms,
                'duration_sec':    duration,
                'accuracy':        metric_map.get('accuracy'),
                'f1_risk':         metric_map.get('f1_risk'),
                'recall_risk':     metric_map.get('recall_risk'),
                'precision_risk':  metric_map.get('precision_risk'),
                'cv_best_f1':      metric_map.get('cv_best_f1'),
                'n_estimators':    param_map.get('n_estimators'),
                'max_depth':       param_map.get('max_depth'),
                'learning_rate':   param_map.get('learning_rate'),
                'quarter':         param_map.get('quarter', '-'),
                'train_size':      param_map.get('train_size'),
            })

        conn.close()
        return result

    except Exception as e:
        return [{'error': str(e)}]


def get_pipeline_log():
    try:
        if not os.path.exists(LOG_PATH):
            return []
        with open(LOG_PATH) as f:
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
        'runs': runs,
        'pipeline_log': log,
        'db_exists': os.path.exists(DB_PATH)
    })
