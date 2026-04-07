"""
ResistanceMap Verification Layer
=================================

Zero-trust verification system for multi-agent resistance prediction.
Every data handoff between agents is cryptographically verified and
statistically validated before trust is granted.

Modules:
  - zero_trust: Core verification infrastructure with hash chains and statistical validation
  - math_reasoning: First-principles mathematical bounds and stability analysis
  - guardrails: Clinical and scientific constraint enforcement

Reference Frameworks:
  - Information Theory: Cover & Thomas (Elements of Information Theory)
  - Dynamical Systems: Strogatz (Nonlinear Dynamics and Chaos)
  - Network Theory: Barabási & Albert (Scale-Free Networks)
  - Cognitive Science: Tversky (Features of Similarity)
  - Cell Biology: Alberts et al. (Molecular Biology of the Cell)
"""

from resistancemap.verification.zero_trust import (
    VerificationResult,
    ZeroTrustVerifier,
)
from resistancemap.verification.math_reasoning import (
    InformationTheoreticBounds,
    DynamicalSystemsTheory,
    NetworkTopologyTheory,
    CognitiveDecisionTheory,
)
from resistancemap.verification.guardrails import (
    Guardrail,
    GuardrailEngine,
)

__all__ = [
    "VerificationResult",
    "ZeroTrustVerifier",
    "InformationTheoreticBounds",
    "DynamicalSystemsTheory",
    "NetworkTopologyTheory",
    "CognitiveDecisionTheory",
    "Guardrail",
    "GuardrailEngine",
]

__version__ = "0.1.0"
