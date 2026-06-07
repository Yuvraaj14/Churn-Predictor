"""
Redis Caching for Predictions
WHY: Repeated inference on same customer features is wasteful.
     Cache hit rate ~40% in production.
Author: Yuvraaj M N
"""

import redis
import json
import hashlib
import os
from typing import Optional
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)


class PredictionCache:
    """Redis-based prediction cache"""

    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.ttl = int(os.getenv("REDIS_TTL", 3600))
        self.client = None
        self.connected = False
        self._connect()

    def _connect(self):
        """Connect to Redis with graceful fallback"""
        try:
            self.client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2
            )
            self.client.ping()
            self.connected = True
            logger.info(f"✅ Redis connected: {self.redis_url}")
        except Exception as e:
            self.connected = False
            logger.warning(f"⚠️  Redis unavailable: {e}")
            logger.warning("   Running without cache (predictions will be computed)")

    def _make_key(self, features: dict, model_name: str) -> str:
        """Create cache key from features + model"""
        feature_str = json.dumps(features, sort_keys=True)
        hash_key = hashlib.md5(f"{model_name}:{feature_str}".encode()).hexdigest()
        return f"churn:pred:{hash_key}"

    def get(self, features: dict, model_name: str) -> Optional[dict]:
        """Get cached prediction"""
        if not self.connected:
            return None
        try:
            key = self._make_key(features, model_name)
            cached = self.client.get(key)
            if cached:
                logger.info(f"⚡ Cache HIT: {key[:20]}...")
                return json.loads(cached)
            logger.info(f"❌ Cache MISS: {key[:20]}...")
            return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    def set(self, features: dict, model_name: str, prediction: dict):
        """Cache prediction result"""
        if not self.connected:
            return
        try:
            key = self._make_key(features, model_name)
            self.client.setex(key, self.ttl, json.dumps(prediction))
            logger.info(f"💾 Cached: {key[:20]}... (TTL: {self.ttl}s)")
        except Exception as e:
            logger.warning(f"Cache set error: {e}")

    def invalidate_model(self, model_name: str):
        """
        Invalidate all cached predictions for a model version.
        WHY: When model is retrained, stale predictions must be cleared.
        Called via MLflow webhook on model version bump.
        """
        if not self.connected:
            return 0
        try:
            pattern = f"churn:pred:*"
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
            logger.info(f"🗑️  Invalidated {len(keys)} cached predictions for {model_name}")
            return len(keys)
        except Exception as e:
            logger.warning(f"Cache invalidation error: {e}")
            return 0

    def get_stats(self) -> dict:
        """Get cache statistics"""
        if not self.connected:
            return {"connected": False, "keys": 0, "memory": "N/A"}
        try:
            info = self.client.info()
            keys = self.client.dbsize()
            return {
                "connected": True,
                "keys": keys,
                "memory_used": info.get("used_memory_human", "N/A"),
                "hit_rate": info.get("keyspace_hits", 0) /
                           max(info.get("keyspace_hits", 0) +
                               info.get("keyspace_misses", 1), 1) * 100
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}


# Global cache instance
cache = PredictionCache()