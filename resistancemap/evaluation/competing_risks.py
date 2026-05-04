"""Competing risks models for intrinsic vs acquired resistance.

Provides cause-specific hazard models and mixture cure models for modeling
patient response trajectories where two competing failure modes (intrinsic
vs acquired resistance) drive outcomes.

References
----------
Tsiatis, A. A. (1975).
    "A Nonidentifiability Aspect of the Problem of Competing Risks."
    PNAS, 72(1), 20-22.
Gaynor, J. J., Feuer, E. J., Tan, C. C., et al. (1993).
    "On the use of cause-specific failure and conditional failure probabilities
    in estimating the risk of death after renal transplantation."
    Statistics in Medicine, 12(12-13), 1091-1100.
Yakovlev, A. Y., & Tsodikov, A. D. (1996).
    "Stochastic Models of Tumor Latency and Their Biostatistical Applications."
    World Scientific.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy import integrate, optimize

logger = logging.getLogger(__name__)

__all__ = [
    "CompetingRisksModel",
    "MixtureCureModel",
    "fit_competing_risks",
    "compute_cif",
]


class CompetingRisksModel:
    """Cause-specific hazard model for two competing risk types.

    Models patient outcomes as arising from two competing failure modes:
    intrinsic resistance (present at baseline) vs acquired resistance
    (develops over time). The mixture classification assigns each patient
    a probability of belonging to each risk type, and separate hazard rates
    govern progression.

    Attributes:
        hazard_intrinsic: Constant hazard rate for intrinsic-resistance patients.
        hazard_acquired: Constant hazard rate for acquired-resistance patients.
        mixture_prob: Probability a given patient is intrinsic-resistance type.
        cif_times: Computed cumulative incidence times.
        cif_intrinsic: Cumulative incidence function (CIF) for intrinsic risk.
        cif_acquired: CIF for acquired risk.
    """

    def __init__(self) -> None:
        """Initialize the competing risks model."""
        self.hazard_intrinsic: float | None = None
        self.hazard_acquired: float | None = None
        self.mixture_prob: float | None = None
        self.cif_times: np.ndarray | None = None
        self.cif_intrinsic: np.ndarray | None = None
        self.cif_acquired: np.ndarray | None = None

    def fit(
        self,
        times: np.ndarray,
        events: np.ndarray,
        risk_types: np.ndarray,
        init_hazard_intrinsic: float = 0.1,
        init_hazard_acquired: float = 0.05,
        init_mixture_prob: float = 0.5,
    ) -> CompetingRisksModel:
        """Fit cause-specific hazard rates and mixture probability.

        Uses maximum likelihood estimation with scipy.optimize.minimize to
        estimate constant hazard rates for each risk type and the fraction
        of patients in the intrinsic-resistance subgroup.

        Args:
            times: (N,) time-to-event or censoring times.
            events: (N,) indicator: 1 if event occurred, 0 if censored.
            risk_types: (N,) cause of event; 0 = intrinsic, 1 = acquired.
                Only meaningful for rows where events == 1.
            init_hazard_intrinsic: Initial guess for intrinsic hazard. Defaults to 0.1.
            init_hazard_acquired: Initial guess for acquired hazard. Defaults to 0.05.
            init_mixture_prob: Initial guess for mixture probability. Defaults to 0.5.

        Returns:
            self (for chaining).

        Raises:
            ValueError: If inputs are invalid.
        """
        times = np.asarray(times, dtype=np.float64)
        events = np.asarray(events, dtype=int)
        risk_types = np.asarray(risk_types, dtype=int)

        if times.ndim != 1 or events.ndim != 1 or risk_types.ndim != 1:
            raise ValueError("times, events, risk_types must be 1D")
        if times.shape[0] != events.shape[0] or times.shape[0] != risk_types.shape[0]:
            raise ValueError("times, events, risk_types length mismatch")
        if np.any(times < 0):
            raise ValueError("times must be non-negative")
        if np.any((events != 0) & (events != 1)):
            raise ValueError("events must be binary (0 or 1)")
        if np.any((risk_types != 0) & (risk_types != 1)):
            raise ValueError("risk_types must be 0 or 1")

        def neg_log_likelihood(params: np.ndarray) -> float:
            h_int, h_acq, p_int = params

            # Clamp to reasonable bounds
            if h_int <= 0 or h_acq <= 0 or not (0 < p_int < 1):
                return 1e10

            nll = 0.0

            # For each observation:
            # If censored: log(S(t)) where S is overall survival
            # If event of type k: log(h_k(t) * S(t))
            for i in range(len(times)):
                t = times[i]
                e = events[i]
                r = risk_types[i]

                # Overall survival: S(t) = exp(-(p*h_int + (1-p)*h_acq)*t)
                overall_hazard = p_int * h_int + (1.0 - p_int) * h_acq
                surv = np.exp(-overall_hazard * t)

                if e == 0:
                    # Censored: log(S(t))
                    nll -= np.log(np.clip(surv, 1e-15, 1.0))
                elif r == 0:
                    # Event: intrinsic risk
                    # log(P(intrinsic) * h_int * S(t))
                    nll -= np.log(np.clip(p_int * h_int * surv, 1e-15, 1.0))
                else:
                    # Event: acquired risk
                    nll -= np.log(np.clip((1.0 - p_int) * h_acq * surv, 1e-15, 1.0))

            return nll

        # Optimize
        x0 = [init_hazard_intrinsic, init_hazard_acquired, init_mixture_prob]
        result = optimize.minimize(
            neg_log_likelihood,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 500},
        )

        self.hazard_intrinsic, self.hazard_acquired, self.mixture_prob = result.x
        logger.info(
            "CompetingRisksModel fitted: h_intrinsic=%.4f, h_acquired=%.4f, p_int=%.4f",
            self.hazard_intrinsic,
            self.hazard_acquired,
            self.mixture_prob,
        )
        return self

    def compute_cif(self, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute cumulative incidence functions for both risk types.

        Args:
            times: (M,) time points to evaluate CIF.

        Returns:
            Tuple of (cif_intrinsic, cif_acquired), each (M,), representing
            the probability of experiencing that cause of failure by time t.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self.hazard_intrinsic is None:
            raise RuntimeError("fit() must be called before compute_cif()")

        times = np.asarray(times, dtype=np.float64)
        cif_int = np.zeros_like(times)
        cif_acq = np.zeros_like(times)

        p_int = self.mixture_prob
        h_int = self.hazard_intrinsic
        h_acq = self.hazard_acquired
        h_total = p_int * h_int + (1.0 - p_int) * h_acq

        for i, t in enumerate(times):
            # CIF for intrinsic: integral from 0 to t of p_int * h_int * S(u) du
            # where S(u) = exp(-h_total * u)
            def integrand_int(u: float) -> float:
                return p_int * h_int * np.exp(-h_total * u)

            def integrand_acq(u: float) -> float:
                return (1.0 - p_int) * h_acq * np.exp(-h_total * u)

            if t > 0:
                cif_int[i], _ = integrate.quad(integrand_int, 0, t, limit=100)
                cif_acq[i], _ = integrate.quad(integrand_acq, 0, t, limit=100)

        return cif_int, cif_acq


class MixtureCureModel:
    """Mixture cure model with fraction of 'cured' (permanently sensitive) patients.

    Models a population with two subgroups:

    - **Cured subgroup** (fraction π): permanently sensitive; never develop resistance.
    - **Uncured subgroup** (fraction 1 - π): develop resistance over time with
      cause-specific hazards for intrinsic vs acquired mechanisms.

    The overall survival curve plateaus at S(∞) = π (the cure fraction).

    Attributes:
        cure_fraction: Fraction π of patients who are cured (permanently sensitive).
        hazard_intrinsic: Hazard rate for intrinsic resistance in uncured subgroup.
        hazard_acquired: Hazard rate for acquired resistance in uncured subgroup.
        mixture_prob: Within uncured subgroup, fraction with intrinsic phenotype.
    """

    def __init__(self) -> None:
        """Initialize the mixture cure model."""
        self.cure_fraction: float | None = None
        self.hazard_intrinsic: float | None = None
        self.hazard_acquired: float | None = None
        self.mixture_prob: float | None = None

    def fit(
        self,
        times: np.ndarray,
        events: np.ndarray,
        risk_types: np.ndarray,
        init_cure: float = 0.3,
        init_hazard_intrinsic: float = 0.1,
        init_hazard_acquired: float = 0.05,
        init_mixture_prob: float = 0.5,
    ) -> MixtureCureModel:
        """Fit cure fraction and cause-specific hazards in uncured subgroup.

        Uses maximum likelihood estimation to estimate all four parameters
        simultaneously.

        Args:
            times: (N,) time-to-event or censoring times.
            events: (N,) indicator: 1 if resistance occurred, 0 if censored.
            risk_types: (N,) cause type (0 = intrinsic, 1 = acquired).
            init_cure: Initial guess for cure fraction. Defaults to 0.3.
            init_hazard_intrinsic: Initial hazard for intrinsic. Defaults to 0.1.
            init_hazard_acquired: Initial hazard for acquired. Defaults to 0.05.
            init_mixture_prob: Initial mixture prob in uncured. Defaults to 0.5.

        Returns:
            self (for chaining).
        """
        times = np.asarray(times, dtype=np.float64)
        events = np.asarray(events, dtype=int)
        risk_types = np.asarray(risk_types, dtype=int)

        if times.shape[0] != events.shape[0] or times.shape[0] != risk_types.shape[0]:
            raise ValueError("times, events, risk_types length mismatch")

        def neg_log_likelihood(params: np.ndarray) -> float:
            cure, h_int, h_acq, p_int = params

            # Bounds check
            if not (0 < cure < 1) or h_int <= 0 or h_acq <= 0 or not (0 < p_int < 1):
                return 1e10

            nll = 0.0
            for i in range(len(times)):
                t = times[i]
                e = events[i]
                r = risk_types[i]

                # Survival for uncured subgroup: S_u(t) = exp(-(p*h_int + (1-p)*h_acq)*t)
                h_uncured = p_int * h_int + (1.0 - p_int) * h_acq
                s_uncured = np.exp(-h_uncured * t)

                # Overall survival: S(t) = cure + (1 - cure) * S_u(t)
                s_overall = cure + (1.0 - cure) * s_uncured

                if e == 0:
                    # Censored
                    nll -= np.log(np.clip(s_overall, 1e-15, 1.0))
                elif r == 0:
                    # Intrinsic resistance: (1 - cure) * p * h_int * S_u(t)
                    nll -= np.log(
                        np.clip(
                            (1.0 - cure) * p_int * h_int * s_uncured, 1e-15, 1.0
                        )
                    )
                else:
                    # Acquired resistance: (1 - cure) * (1-p) * h_acq * S_u(t)
                    nll -= np.log(
                        np.clip(
                            (1.0 - cure) * (1.0 - p_int) * h_acq * s_uncured,
                            1e-15,
                            1.0,
                        )
                    )

            return nll

        x0 = [init_cure, init_hazard_intrinsic, init_hazard_acquired, init_mixture_prob]
        result = optimize.minimize(
            neg_log_likelihood,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 500},
        )

        (
            self.cure_fraction,
            self.hazard_intrinsic,
            self.hazard_acquired,
            self.mixture_prob,
        ) = result.x
        logger.info(
            "MixtureCureModel fitted: cure=%.4f, h_int=%.4f, h_acq=%.4f, p_int=%.4f",
            self.cure_fraction,
            self.hazard_intrinsic,
            self.hazard_acquired,
            self.mixture_prob,
        )
        return self

    def survival_function(self, times: np.ndarray) -> np.ndarray:
        """Compute overall survival S(t) = cure + (1 - cure) * S_u(t).

        Args:
            times: (M,) time points.

        Returns:
            (M,) survival probabilities.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self.cure_fraction is None:
            raise RuntimeError("fit() must be called before survival_function()")

        times = np.asarray(times, dtype=np.float64)
        h_uncured = (
            self.mixture_prob * self.hazard_intrinsic
            + (1.0 - self.mixture_prob) * self.hazard_acquired
        )
        s_uncured = np.exp(-h_uncured * times)
        s_overall = self.cure_fraction + (1.0 - self.cure_fraction) * s_uncured
        return s_overall


def fit_competing_risks(
    times: np.ndarray,
    events: np.ndarray,
    risk_types: np.ndarray,
    covariates: np.ndarray | None = None,
) -> CompetingRisksModel:
    """Convenience function to fit a competing risks model.

    Currently ignores covariates (constant hazards assumed). Future versions
    may support proportional hazards with covariate adjustment.

    Args:
        times: (N,) time-to-event times.
        events: (N,) binary event indicator.
        risk_types: (N,) cause-specific risk type (0 or 1).
        covariates: Optional (N, P) covariate matrix (currently unused).

    Returns:
        Fitted CompetingRisksModel instance.
    """
    if covariates is not None:
        logger.warning(
            "fit_competing_risks: covariates provided but not yet supported; "
            "constant hazards assumed"
        )

    model = CompetingRisksModel()
    model.fit(times, events, risk_types)
    return model


def compute_cif(
    hazard_rates: dict[str, float],
    times: np.ndarray,
) -> dict[str, np.ndarray]:
    """Convenience function to compute CIF from given hazard rates.

    Assumes two competing risks with constant hazards and a mixture model.

    Args:
        hazard_rates: Dictionary with keys:

            - ``"intrinsic"`` (float): Hazard for intrinsic resistance.
            - ``"acquired"`` (float): Hazard for acquired resistance.
            - ``"mixture_prob"`` (float): Probability of intrinsic type.

        times: (M,) time points to evaluate CIF.

    Returns:
        Dictionary with keys:

        - ``"times"`` (ndarray): Echoed input times.
        - ``"cif_intrinsic"`` (ndarray): CIF for intrinsic risk.
        - ``"cif_acquired"`` (ndarray): CIF for acquired risk.
    """
    h_int = hazard_rates["intrinsic"]
    h_acq = hazard_rates["acquired"]
    p_int = hazard_rates["mixture_prob"]

    times = np.asarray(times, dtype=np.float64)
    h_total = p_int * h_int + (1.0 - p_int) * h_acq

    cif_int = np.zeros_like(times)
    cif_acq = np.zeros_like(times)

    for i, t in enumerate(times):
        if t > 0:
            def integrand_int(u: float) -> float:
                return p_int * h_int * np.exp(-h_total * u)

            def integrand_acq(u: float) -> float:
                return (1.0 - p_int) * h_acq * np.exp(-h_total * u)

            cif_int[i], _ = integrate.quad(integrand_int, 0, t, limit=100)
            cif_acq[i], _ = integrate.quad(integrand_acq, 0, t, limit=100)

    return {
        "times": times,
        "cif_intrinsic": cif_int,
        "cif_acquired": cif_acq,
    }
