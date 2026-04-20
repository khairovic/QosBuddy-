"""
QoSBuddy M6 — DSO1.1 QoS Estimation integration
================================================
Surfaces Member 2's jitter/throughput forecasts to the serving layer:
  • predicted_jitter.csv — XGBoost predictions (y_true, y_pred per row)
  • forecast_ci.csv      — Prophet forecast with confidence intervals

The XGBoost / LSTM models live under ``qos_forecast/models/`` and are
available for future live-inference work; today the API serves the
pre-computed predictions for dashboard panels.
"""
