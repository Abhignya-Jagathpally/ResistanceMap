"""
MCP server connectors for ResistanceMap.

Provides abstractions for connecting to external data sources via MCP:
- PubMed literature search and validation
- ChEMBL drug-target information
- ClinicalTrials.gov trial data
- HuggingFace model hub for ESM-2 embeddings

Each connector implements the MCPConnector interface for unified async access,
with built-in caching and rate limiting.
"""

import asyncio
import logging
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict
import heapq

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entry in the MCP response cache."""

    value: Any
    timestamp: datetime
    ttl_seconds: int

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.utcnow() - self.timestamp > timedelta(seconds=self.ttl_seconds)


class LRUCache:
    """Least Recently Used cache with TTL support."""

    def __init__(self, max_size: int = 1000):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of entries before eviction
        """
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: OrderedDict = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, updating access order."""
        if key not in self._cache:
            return None

        entry = self._cache[key]
        if entry.is_expired():
            del self._cache[key]
            if key in self._access_order:
                del self._access_order[key]
            return None

        # Update access order
        self._access_order.move_to_end(key)
        return entry.value

    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        """Set value in cache with TTL."""
        if len(self._cache) >= self.max_size and key not in self._cache:
            # Evict least recently used
            lru_key = next(iter(self._access_order))
            del self._cache[lru_key]
            del self._access_order[lru_key]

        self._cache[key] = CacheEntry(value, datetime.utcnow(), ttl_seconds)
        self._access_order[key] = True
        self._access_order.move_to_end(key)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, rate: float, burst: int = 1):
        """
        Initialize rate limiter.

        Args:
            rate: Tokens per second
            burst: Maximum burst size
        """
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)
        self.last_update = datetime.utcnow()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens, blocking if necessary.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            Time waited in seconds
        """
        async with self._lock:
            now = datetime.utcnow()
            time_passed = (now - self.last_update).total_seconds()
            self.tokens = min(self.burst, self.tokens + time_passed * self.rate)
            self.last_update = now

            if self.tokens < tokens:
                wait_time = (tokens - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
                self.last_update = datetime.utcnow()
            else:
                self.tokens -= tokens

            return max(0, wait_time) if self.tokens < tokens else 0


class MCPConnector(ABC):
    """
    Abstract base class for MCP server connections.

    All connectors should implement async query methods and response validation.
    """

    def __init__(self, name: str, rate_limit: float = 10.0):
        """
        Initialize MCP connector.

        Args:
            name: Connector name (e.g., "pubmed", "chembl")
            rate_limit: Requests per second allowed
        """
        self.name = name
        self.rate_limiter = RateLimiter(rate=rate_limit, burst=max(1, int(rate_limit)))
        self.cache = LRUCache(max_size=500)
        self.logger = logger

    @abstractmethod
    async def query(self, params: dict) -> dict:
        """
        Execute a query against the MCP server.

        Args:
            params: Query parameters specific to the connector

        Returns:
            Query result
        """
        pass

    @abstractmethod
    def validate_response(self, response: dict) -> Tuple[bool, Optional[str]]:
        """
        Validate MCP server response.

        Args:
            response: Response from MCP server

        Returns:
            Tuple of (is_valid, error_message)
        """
        pass

    def _get_cache_key(self, **kwargs) -> str:
        """Generate cache key from parameters."""
        key_str = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    async def query_with_cache(self, **params) -> Optional[dict]:
        """Execute query with caching."""
        cache_key = self._get_cache_key(**params)

        # Check cache
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.logger.debug(f"{self.name}: Cache hit for {cache_key[:8]}")
            return cached

        # Rate limit
        await self.rate_limiter.acquire(1)

        # Execute query
        try:
            result = await self.query(params)
            is_valid, error_msg = self.validate_response(result)

            if not is_valid:
                self.logger.error(f"{self.name}: Invalid response - {error_msg}")
                return None

            # Cache successful response
            self.cache.set(cache_key, result, ttl_seconds=3600)
            return result

        except Exception as e:
            self.logger.error(f"{self.name}: Query failed - {e}")
            return None


class PubMedConnector(MCPConnector):
    """
    Connector to PubMed MCP for literature search and validation.

    Provides methods for:
    - Searching drug resistance literature
    - Validating protein-drug associations
    - Retrieving recent MM research
    """

    def __init__(self, rate_limit: float = 3.0):
        """Initialize PubMed connector with conservative rate limit."""
        super().__init__("pubmed", rate_limit=rate_limit)

    async def query(self, params: dict) -> dict:
        """
        Execute PubMed search query.

        Args:
            params: Query parameters
                - query: Search query string
                - max_results: Maximum results to return
                - date_from/date_to: Date range filters

        Returns:
            Query results with articles
        """
        # Placeholder for actual MCP call
        # In production, this would call: mcp_server.search_articles(...)
        query_str = params.get("query", "")
        max_results = params.get("max_results", 20)

        self.logger.info(f"PubMed search: '{query_str[:50]}...' (max: {max_results})")

        # Simulate response
        return {
            "query": query_str,
            "total_results": 0,
            "articles": [],
            "status": "success",
        }

    def validate_response(self, response: dict) -> Tuple[bool, Optional[str]]:
        """Validate PubMed response."""
        required_fields = ["query", "total_results", "articles", "status"]
        if not all(field in response for field in required_fields):
            return False, "Missing required response fields"

        if response["status"] != "success":
            return False, f"Query failed with status: {response['status']}"

        if not isinstance(response["articles"], list):
            return False, "Articles field must be a list"

        return True, None

    async def search_resistance_literature(
        self,
        gene: str,
        drug: str,
        max_results: int = 50,
    ) -> List[dict]:
        """
        Search for literature on drug resistance to specific gene-drug combinations.

        Args:
            gene: Gene/protein name
            drug: Drug name
            max_results: Maximum articles to return

        Returns:
            List of article summaries
        """
        query = f"{gene} {drug} resistance multiple myeloma"
        result = await self.query_with_cache(
            query=query,
            max_results=max_results,
        )

        if result is None:
            return []

        return result.get("articles", [])

    async def verify_protein_drug_association(
        self,
        protein: str,
        drug: str,
    ) -> dict:
        """
        Verify protein-drug association in literature.

        Args:
            protein: Protein name
            drug: Drug name

        Returns:
            Association confidence and supporting articles
        """
        query = f"{protein} {drug} binding mechanism interaction"
        result = await self.query_with_cache(
            query=query,
            max_results=10,
        )

        if result is None:
            return {
                "protein": protein,
                "drug": drug,
                "confidence": 0.0,
                "evidence_count": 0,
                "articles": [],
            }

        articles = result.get("articles", [])
        confidence = min(1.0, len(articles) * 0.1)

        return {
            "protein": protein,
            "drug": drug,
            "confidence": float(confidence),
            "evidence_count": len(articles),
            "articles": articles,
        }

    async def get_latest_mm_studies(self, n: int = 50) -> List[dict]:
        """
        Retrieve latest multiple myeloma research studies.

        Args:
            n: Number of studies to retrieve

        Returns:
            List of recent MM studies
        """
        query = "multiple myeloma treatment outcome"
        result = await self.query_with_cache(
            query=query,
            max_results=n,
        )

        if result is None:
            return []

        return result.get("articles", [])


class ChEMBLConnector(MCPConnector):
    """
    Connector to ChEMBL MCP for drug-target validation.

    Provides methods for:
    - Retrieving drug targets
    - Getting bioactivity data
    - Validating predicted protein-drug interactions
    """

    def __init__(self, rate_limit: float = 5.0):
        """Initialize ChEMBL connector."""
        super().__init__("chembl", rate_limit=rate_limit)

    async def query(self, params: dict) -> dict:
        """
        Execute ChEMBL query.

        Args:
            params: Query parameters
                - compound_name: Drug/compound name
                - target_id: ChEMBL target ID
                - bioactivity_type: IC50, Ki, Kd, EC50, etc.

        Returns:
            Query results
        """
        # Placeholder for actual MCP call
        # In production: mcp_server.compound_search(...) or get_bioactivity(...)
        self.logger.info(f"ChEMBL query: {params}")

        return {
            "query": params,
            "results": [],
            "status": "success",
        }

    def validate_response(self, response: dict) -> Tuple[bool, Optional[str]]:
        """Validate ChEMBL response."""
        required_fields = ["query", "results", "status"]
        if not all(field in response for field in required_fields):
            return False, "Missing required response fields"

        if response["status"] != "success":
            return False, f"Query failed with status: {response['status']}"

        return True, None

    async def get_drug_targets(self, drug_name: str) -> List[dict]:
        """
        Get targets for a drug.

        Args:
            drug_name: Name of drug

        Returns:
            List of targets with bioactivity data
        """
        result = await self.query_with_cache(
            compound_name=drug_name,
        )

        if result is None:
            return []

        return result.get("results", [])

    async def get_bioactivity(self, target_chembl_id: str, activity_type: str = "IC50") -> List[dict]:
        """
        Get bioactivity data for a target.

        Args:
            target_chembl_id: ChEMBL target ID
            activity_type: Type of bioactivity (IC50, Ki, etc.)

        Returns:
            List of bioactivity measurements
        """
        result = await self.query_with_cache(
            target_id=target_chembl_id,
            bioactivity_type=activity_type,
        )

        if result is None:
            return []

        return result.get("results", [])

    async def validate_predicted_target(
        self,
        protein: str,
        drug: str,
    ) -> dict:
        """
        Validate predicted protein-drug interaction using ChEMBL.

        Args:
            protein: Protein name
            drug: Drug name

        Returns:
            Validation results with confidence score
        """
        # Try to find drug in ChEMBL
        drug_results = await self.get_drug_targets(drug)

        if not drug_results:
            return {
                "protein": protein,
                "drug": drug,
                "is_valid": False,
                "confidence": 0.0,
                "evidence_count": 0,
            }

        # Check if protein is in target list
        matching_targets = [
            r for r in drug_results
            if protein.upper() in r.get("target_name", "").upper()
        ]

        confidence = min(1.0, len(matching_targets) * 0.3)

        return {
            "protein": protein,
            "drug": drug,
            "is_valid": len(matching_targets) > 0,
            "confidence": float(confidence),
            "evidence_count": len(matching_targets),
            "matching_targets": matching_targets,
        }


class ClinicalTrialsConnector(MCPConnector):
    """
    Connector to ClinicalTrials.gov MCP for trial data.

    Provides methods for:
    - Searching MM trials
    - Retrieving trial outcomes
    - Identifying drug-protein pairs in clinical validation
    """

    def __init__(self, rate_limit: float = 5.0):
        """Initialize ClinicalTrials connector."""
        super().__init__("clinical_trials", rate_limit=rate_limit)

    async def query(self, params: dict) -> dict:
        """
        Execute ClinicalTrials.gov query.

        Args:
            params: Query parameters
                - condition: Disease condition
                - intervention: Drug or intervention name
                - phase: Trial phase
                - status: Recruitment status

        Returns:
            Query results
        """
        # Placeholder for actual MCP call
        # In production: mcp_server.search_trials(...)
        self.logger.info(f"ClinicalTrials query: {params}")

        return {
            "query": params,
            "trials": [],
            "status": "success",
        }

    def validate_response(self, response: dict) -> Tuple[bool, Optional[str]]:
        """Validate ClinicalTrials response."""
        required_fields = ["query", "trials", "status"]
        if not all(field in response for field in required_fields):
            return False, "Missing required response fields"

        if not isinstance(response["trials"], list):
            return False, "Trials field must be a list"

        return True, None

    async def search_mm_trials(
        self,
        drug: str,
        phase: Optional[str] = None,
    ) -> List[dict]:
        """
        Search for multiple myeloma trials of a specific drug.

        Args:
            drug: Drug name
            phase: Trial phase filter (e.g., "PHASE3")

        Returns:
            List of trials
        """
        result = await self.query_with_cache(
            condition="multiple myeloma",
            intervention=drug,
            phase=phase,
        )

        if result is None:
            return []

        return result.get("trials", [])

    async def get_trial_outcomes(self, nct_id: str) -> dict:
        """
        Get outcomes for a specific trial.

        Args:
            nct_id: NCT trial identifier

        Returns:
            Trial outcomes and efficacy data
        """
        # In production, would call: mcp_server.get_trial_details(nct_id)
        return {
            "nct_id": nct_id,
            "primary_outcomes": [],
            "secondary_outcomes": [],
            "results_available": False,
        }


class HuggingFaceConnector(MCPConnector):
    """
    Connector to HuggingFace model hub for ESM-2 embeddings.

    Provides methods for:
    - Downloading ESM-2 models
    - Checking model versions
    - Streaming model weights
    """

    def __init__(self, rate_limit: float = 10.0):
        """Initialize HuggingFace connector."""
        super().__init__("huggingface", rate_limit=rate_limit)

    async def query(self, params: dict) -> dict:
        """
        Execute HuggingFace query.

        Args:
            params: Query parameters
                - model_name: HF model identifier
                - action: "download", "info", "list"

        Returns:
            Query results
        """
        # Placeholder for actual HF API call
        self.logger.info(f"HuggingFace query: {params}")

        return {
            "model": params.get("model_name", ""),
            "status": "success",
        }

    def validate_response(self, response: dict) -> Tuple[bool, Optional[str]]:
        """Validate HuggingFace response."""
        if "status" not in response:
            return False, "Missing status field"

        if response["status"] != "success":
            return False, f"Query failed with status: {response['status']}"

        return True, None

    async def download_esm2_model(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        cache_dir: str = "./models",
    ) -> str:
        """
        Download ESM-2 model for protein embeddings.

        Args:
            model_name: HuggingFace model identifier
            cache_dir: Local cache directory

        Returns:
            Path to downloaded model
        """
        result = await self.query_with_cache(
            model_name=model_name,
            action="download",
            cache_dir=cache_dir,
        )

        if result is None or not result.get("status") == "success":
            self.logger.error(f"Failed to download {model_name}")
            return ""

        return f"{cache_dir}/{model_name.split('/')[-1]}"

    async def check_model_version(self, model_name: str) -> dict:
        """
        Check latest version of a model.

        Args:
            model_name: HuggingFace model identifier

        Returns:
            Model metadata and version info
        """
        result = await self.query_with_cache(
            model_name=model_name,
            action="info",
        )

        if result is None:
            return {
                "model": model_name,
                "available": False,
                "version": None,
            }

        return {
            "model": model_name,
            "available": result.get("status") == "success",
            "version": result.get("version"),
        }


class MCPOrchestrator:
    """
    Manages all MCP connections with coordinated access.

    Handles:
    - Connector lifecycle
    - Cross-connector caching
    - Global rate limiting
    - Error handling and retries
    """

    def __init__(self):
        """Initialize MCP orchestrator."""
        self.connectors: Dict[str, MCPConnector] = {}
        self._cache: LRUCache = LRUCache(max_size=2000)
        self._rate_limiters: Dict[str, RateLimiter] = {}
        self.logger = logger

    def register_connector(self, connector: MCPConnector) -> None:
        """
        Register an MCP connector.

        Args:
            connector: Connector instance to register
        """
        self.connectors[connector.name] = connector
        self.logger.info(f"Registered connector: {connector.name}")

    async def query(
        self,
        connector_name: str,
        method: str,
        params: dict,
        use_cache: bool = True,
    ) -> Optional[dict]:
        """
        Execute a query on a specific connector.

        Args:
            connector_name: Name of connector to use
            method: Method name on connector
            params: Query parameters
            use_cache: Whether to use global cache

        Returns:
            Query result or None on failure
        """
        if connector_name not in self.connectors:
            self.logger.error(f"Unknown connector: {connector_name}")
            return None

        connector = self.connectors[connector_name]

        # Check global cache
        if use_cache:
            cache_key = f"{connector_name}:{method}:{json.dumps(params, sort_keys=True, default=str)}"
            cached = self._cache.get(cache_key)
            if cached is not None:
                self.logger.debug(f"Global cache hit for {cache_key[:30]}")
                return cached

        # Execute query
        try:
            method_fn = getattr(connector, method)
            result = await method_fn(**params)

            # Cache result
            if use_cache and result is not None:
                self._cache.set(cache_key, result, ttl_seconds=3600)

            return result
        except AttributeError:
            self.logger.error(f"Unknown method: {connector_name}.{method}")
            return None
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return None

    def get_cached(self, key: str) -> Optional[Any]:
        """Get value from global cache."""
        return self._cache.get(key)

    def set_cached(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        """Set value in global cache."""
        self._cache.set(key, value, ttl_seconds)

    def clear_cache(self, connector_name: Optional[str] = None) -> None:
        """
        Clear cache entries.

        Args:
            connector_name: If specified, only clear that connector's cache
        """
        if connector_name is None:
            self._cache.clear()
        else:
            # Clear entries for specific connector
            keys_to_delete = [
                k for k in self._cache._cache.keys()
                if k.startswith(f"{connector_name}:")
            ]
            for k in keys_to_delete:
                del self._cache._cache[k]
                if k in self._cache._access_order:
                    del self._cache._access_order[k]

    def cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "total_entries": self._cache.size(),
            "max_size": self._cache.max_size,
        }
