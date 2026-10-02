"""
Continuous Model Evaluation & Retraining Governance for SkyGuard AI.
Compares model predictions against reviewed and verified events,
tracks false alarms and missed anomalies, and retrains models ONLY when validated data is available.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from database.db import (
    record_model_evaluation,
    get_model_evaluation_metrics,
    get_model_evaluations,
    execute_gated_retraining,
    get_retraining_history
)


class ContinuousModelEvaluator:
    """
    Manages continuous operational evaluation for SkyGuard AI.
    - Tracks expert reviews: True Positives, False Alarms, Missed Anomalies.
    - Computes real-time precision, recall, false alarm rate, and missed anomaly rate.
    - Enforces gating: Retraining models ONLY when human-validated ground-truth data is available.
    """

    def __init__(self, min_validated_threshold: int = 20):
        self.min_validated_threshold = min_validated_threshold

    def review_event(
        self,
        event_id: str,
        station_id: str,
        model_prediction: str,
        model_confidence: float,
        verified_status: str,
        reviewed_by: str,
        review_notes: str = "",
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submits an operator / meteorologist verification review for a detected event or telemetry frame.
        verified_status:
        - 'VERIFIED_TRUE_POSITIVE': Confirmed true sensor anomaly or fault
        - 'VERIFIED_FALSE_ALARM': Model falsely flagged genuine weather / nominal reading
        - 'VERIFIED_MISSED_ANOMALY': Engineer detected an anomaly that model failed to catch
        - 'VERIFIED_GENUINE_WEATHER': Model correctly recognized genuine meteorological event
        """
        ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        res = record_model_evaluation(
            event_id=event_id,
            station_id=station_id,
            timestamp=ts,
            model_prediction=model_prediction,
            model_confidence=model_confidence,
            verified_status=verified_status,
            reviewed_by=reviewed_by,
            review_notes=review_notes,
            is_validated_for_training=True
        )
        # Return updated metrics after review
        metrics = self.get_evaluation_metrics()
        return {
            "success": True,
            "review": res,
            "updated_metrics": metrics
        }

    def get_evaluation_metrics(self) -> Dict[str, Any]:
        """Returns live evaluation metrics comparing model predictions against ground truth."""
        return get_model_evaluation_metrics()

    def get_review_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves history of reviewed and verified events."""
        return get_model_evaluations(limit=limit)

    def trigger_gated_retraining(self, triggered_by: str) -> Dict[str, Any]:
        """
        Executes model retraining ONLY when validated data meets the threshold.
        Prevents poisoning models with unverified noisy inputs.
        """
        return execute_gated_retraining(
            triggered_by=triggered_by,
            min_validated_threshold=self.min_validated_threshold
        )

    def get_retraining_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieves history of model retraining runs."""
        return get_retraining_history(limit=limit)
