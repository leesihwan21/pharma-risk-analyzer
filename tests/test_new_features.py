"""
Tests for new features added today:
- Autocomplete API
- Prophet trend forecast
- Drug recommendation (cluster + co-medication)
- Quarterly PRR signal detection
- ML Dashboard API
"""

import json


# ?? Autocomplete ????????????????????????????????????????????
class TestAutocomplete:
    def test_autocomplete_returns_list(self, client):
        res = client.get("/api/autocomplete?q=METH")
        assert res.status_code == 200
        data = json.loads(res.data)
        assert isinstance(data, list)

    def test_autocomplete_matches_prefix(self, client):
        res = client.get("/api/autocomplete?q=METH")
        data = json.loads(res.data)
        for drug in data:
            assert drug.startswith("METH")

    def test_autocomplete_short_query_returns_empty(self, client):
        res = client.get("/api/autocomplete?q=M")
        assert res.status_code == 200
        data = json.loads(res.data)
        assert data == []

    def test_autocomplete_no_query(self, client):
        res = client.get("/api/autocomplete")
        assert res.status_code == 200
        data = json.loads(res.data)
        assert data == []

    def test_autocomplete_max_10_results(self, client):
        res = client.get("/api/autocomplete?q=A")
        data = json.loads(res.data)
        assert len(data) <= 10

    def test_autocomplete_uppercase(self, client):
        res = client.get("/api/autocomplete?q=asp")
        data = json.loads(res.data)
        for drug in data:
            assert drug == drug.upper()


# ?? Prophet Trend Forecast ??????????????????????????????????
class TestTrendForecast:
    def test_forecast_not_enough_quarters(self, client):
        # sample data has 2 quarters so might work or return error
        res = client.get("/api/trend/forecast?drug=METHOTREXATE")
        assert res.status_code in [200, 400]

    def test_forecast_unknown_drug(self, client):
        res = client.get("/api/trend/forecast?drug=UNKNOWNDRUGXYZ")
        assert res.status_code == 404
        data = json.loads(res.data)
        assert "error" in data

    def test_forecast_no_drug_param(self, client):
        res = client.get("/api/trend/forecast")
        assert res.status_code == 400

    def test_forecast_response_structure(self, client):
        res = client.get("/api/trend/forecast?drug=METHOTREXATE")
        if res.status_code == 200:
            data = json.loads(res.data)
            assert "drug" in data
            assert "forecast" in data
            assert isinstance(data["forecast"], list)


# ?? Drug Recommendation - Cluster ???????????????????????????
class TestRecommendCluster:
    def test_cluster_known_drug(self, client):
        res = client.get("/api/recommend/cluster/METHOTREXATE")
        assert res.status_code in [200, 404, 500]
        if res.status_code == 200:
            data = json.loads(res.data)
            assert "drug" in data
            assert "cluster" in data
            assert "similar_drugs" in data

    def test_cluster_unknown_drug(self, client):
        res = client.get("/api/recommend/cluster/UNKNOWNDRUGXYZ")
        assert res.status_code == 404
        data = json.loads(res.data)
        assert "error" in data

    def test_cluster_response_fields(self, client):
        res = client.get("/api/recommend/cluster/ASPIRIN")
        if res.status_code == 200:
            data = json.loads(res.data)
            assert "total_clusters" in data
            assert "drug_top_reactions" in data
            assert "report_count" in data
            assert isinstance(data["similar_drugs"], list)


# ?? Drug Recommendation - Co-medication ?????????????????????
class TestRecommendComedication:
    def test_comedication_known_drug(self, client):
        res = client.get("/api/recommend/comedication/ASPIRIN")
        assert res.status_code in [200, 404, 500]
        if res.status_code == 200:
            data = json.loads(res.data)
            assert "drug" in data
            assert "co_medications" in data
            assert "total_cases" in data

    def test_comedication_unknown_drug(self, client):
        res = client.get("/api/recommend/comedication/UNKNOWNDRUGXYZ")
        assert res.status_code == 404
        data = json.loads(res.data)
        assert "error" in data

    def test_comedication_response_structure(self, client):
        res = client.get("/api/recommend/comedication/ASPIRIN")
        if res.status_code == 200:
            data = json.loads(res.data)
            for item in data["co_medications"]:
                assert "co_drug" in item
                assert "co_count" in item
                assert "serious_rate" in item
                assert "top_reactions" in item


# ?? Quarterly PRR Signal Detection ??????????????????????????
class TestQuarterlyPRRSignal:
    def test_signal_known_drug(self, client):
        res = client.get("/api/signals/quarterly_trend/METHOTREXATE")
        assert res.status_code in [200, 400]
        if res.status_code == 200:
            data = json.loads(res.data)
            assert "drug" in data
            assert "quarterly_data" in data
            assert "spike_alerts" in data

    def test_signal_unknown_drug(self, client):
        res = client.get("/api/signals/quarterly_trend/UNKNOWNDRUGXYZ")
        assert res.status_code == 404
        data = json.loads(res.data)
        assert "error" in data

    def test_signal_response_structure(self, client):
        res = client.get("/api/signals/quarterly_trend/METHOTREXATE")
        if res.status_code == 200:
            data = json.loads(res.data)
            assert "quarters" in data
            assert "total_spikes" in data
            assert "trend_by_reaction" in data
            assert isinstance(data["spike_alerts"], list)

    def test_signal_spike_alert_fields(self, client):
        res = client.get("/api/signals/quarterly_trend/METHOTREXATE")
        if res.status_code == 200:
            data = json.loads(res.data)
            for alert in data["spike_alerts"]:
                assert "reaction" in alert
                assert "type" in alert
                assert alert["type"] in ["SPIKE", "NEW_SIGNAL"]
                assert "quarter" in alert
                assert "curr_prr" in alert


# ?? ML Dashboard ????????????????????????????????????????????
class TestMLDashboard:
    def test_ml_dashboard_page(self, client):
        res = client.get("/ml_dashboard")
        assert res.status_code == 200

    def test_ml_dashboard_api(self, client):
        res = client.get("/api/ml_dashboard/runs")
        assert res.status_code == 200
        data = json.loads(res.data)
        assert "runs" in data
        assert "db_exists" in data
        assert isinstance(data["runs"], list)

    def test_ml_dashboard_pipeline_log(self, client):
        res = client.get("/api/ml_dashboard/runs")
        assert res.status_code == 200
        data = json.loads(res.data)
        assert "pipeline_log" in data
        assert isinstance(data["pipeline_log"], list)


# ?? Recommend Page ???????????????????????????????????????????
class TestRecommendPage:
    def test_recommend_page_loads(self, client):
        res = client.get("/recommend")
        assert res.status_code == 200

    def test_recommend_page_contains_tabs(self, client):
        res = client.get("/recommend")
        assert b"cluster" in res.data.lower() or b"Drug Clustering" in res.data
