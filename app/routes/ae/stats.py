"""
app/routes/ae/stats.py
AE 통계 대시보드
"""
from flask import jsonify

from . import ae
from app.models import AEReport


@ae.route('/api/ae/stats')
def ae_stats():
    reports = AEReport.query.all()
    if not reports:
        return jsonify({'message': '등록된 AE가 없습니다'})

    grade_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    causality_dist = {}
    outcome_dist = {}
    for r in reports:
        if r.ctcae_grade:
            grade_dist[r.ctcae_grade] = grade_dist.get(r.ctcae_grade, 0) + 1
        if r.causality:
            causality_dist[r.causality] = causality_dist.get(r.causality, 0) + 1
        if r.outcome:
            outcome_dist[r.outcome] = outcome_dist.get(r.outcome, 0) + 1

    return jsonify({
        'total': len(reports),
        'sae_count': sum(1 for r in reports if r.is_sae),
        'submitted_count': sum(1 for r in reports if r.is_submitted),
        'grade_distribution': grade_dist,
        'causality_distribution': causality_dist,
        'outcome_distribution': outcome_dist,
    })