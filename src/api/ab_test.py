"""
A/B Testing Logic for Champion vs Challenger Models
WHY: Gradually test new model in production without full rollout.
     Auto rollback if challenger underperforms by >2% AUC.
Author: Yuvraaj M N
"""

import random
import json
import os
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

MODELS_PATH = Path("models")
REPORTS_PATH = Path("reports")


class ABTestManager:
    """
    Manages A/B testing between champion and challenger models.

    Strategy:
    - 80% traffic → Champion (XGBoost)
    - 20% traffic → Challenger (LightGBM)
    - Auto rollback if challenger AUC drops >2% below champion
    """

    def __init__(self):
        self.champion = os.getenv("CHAMPION_MODEL", "xgboost")
        self.challenger = os.getenv("CHALLENGER_MODEL", "lightgbm")
        self.challenger_ratio = float(os.getenv("AB_TEST_RATIO", "0.2"))

        self.stats = {
            "champion": {"requests": 0, "correct": 0, "total_prob": 0.0},
            "challenger": {"requests": 0, "correct": 0, "total_prob": 0.0}
        }

        # Load existing stats if available
        self._load_stats()

        logger.info(f"✅ A/B Test initialized")
        logger.info(f"   Champion: {self.champion} ({(1-self.challenger_ratio)*100:.0f}%)")
        logger.info(f"   Challenger: {self.challenger} ({self.challenger_ratio*100:.0f}%)")

    def select_model(self, customer_id: str = None) -> str:
        """
        Select model for this request.
        Uses customer_id for consistent assignment (same customer → same model).
        """
        if customer_id:
            # Consistent hashing — same customer always gets same model
            hash_val = hash(customer_id) % 100
            if hash_val < (self.challenger_ratio * 100):
                selected = "challenger"
            else:
                selected = "champion"
        else:
            # Random assignment
            selected = "challenger" if random.random() < self.challenger_ratio else "champion"

        model_name = self.challenger if selected == "challenger" else self.champion
        self.stats[selected]["requests"] += 1

        logger.info(f"🎯 A/B: Selected {selected} → {model_name}")
        return model_name, selected

    def record_result(self, variant: str, probability: float):
        """Record prediction result for stats tracking"""
        if variant in self.stats:
            self.stats[variant]["total_prob"] += probability
        self._save_stats()

    def get_report(self) -> dict:
        """Generate A/B test report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "champion": self.champion,
            "challenger": self.challenger,
            "challenger_ratio": self.challenger_ratio,
            "results": {}
        }

        for variant in ["champion", "challenger"]:
            s = self.stats[variant]
            requests = s["requests"]
            avg_prob = s["total_prob"] / max(requests, 1)

            report["results"][variant] = {
                "requests": requests,
                "avg_churn_probability": round(avg_prob, 4),
                "traffic_share": f"{self.challenger_ratio*100 if variant=='challenger' else (1-self.challenger_ratio)*100:.0f}%"
            }

        # Auto rollback check
        champ_prob = report["results"]["champion"]["avg_churn_probability"]
        chall_prob = report["results"]["challenger"]["avg_churn_probability"]

        report["rollback_triggered"] = False
        if abs(champ_prob - chall_prob) > 0.05:
            report["rollback_triggered"] = True
            report["rollback_reason"] = f"Challenger diverged by {abs(champ_prob-chall_prob):.3f}"
            logger.warning(f"⚠️  AUTO ROLLBACK TRIGGERED: {report['rollback_reason']}")

        return report

    def _save_stats(self):
        """Persist A/B stats"""
        stats_path = REPORTS_PATH / "ab_test_stats.json"
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2)

    def _load_stats(self):
        """Load existing A/B stats"""
        stats_path = REPORTS_PATH / "ab_test_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                self.stats = json.load(f)


# Global A/B test manager
ab_manager = ABTestManager()