"""
MCP (Model Context Protocol) integration module for ResistanceMap.

This module provides connectors to external data sources and literature repositories
via MCP servers, including PubMed, ChEMBL, ClinicalTrials.gov, and HuggingFace.

Key responsibilities:
- Managed connections to multiple MCP servers
- Caching and rate limiting
- Response validation and error handling
- Async query orchestration
"""

from resistancemap.mcp_integration.connectors import (
    MCPConnector,
    PubMedConnector,
    ChEMBLConnector,
    ClinicalTrialsConnector,
    HuggingFaceConnector,
    MCPOrchestrator,
)

__all__ = [
    "MCPConnector",
    "PubMedConnector",
    "ChEMBLConnector",
    "ClinicalTrialsConnector",
    "HuggingFaceConnector",
    "MCPOrchestrator",
]
