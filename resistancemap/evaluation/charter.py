"""Scientific charter for ResistanceMap evaluation governance.

Encodes the three testable claims that the ResistanceMap pipeline is
*allowed* to make and the operational tests that would falsify each one.
The charter is the contract between the science and the evaluation layer:
nothing is graded against criteria that do not appear here, and every claim
must spell out what evidence would refute it.

Epistemic note
--------------
The marketing phrase "predict resistance before it happens" is, in the
strictest sense, only demonstrable on **prospective or strictly time-ordered
data**. Cross-sectional snapshots (CCLE, GDSC, single-time-point scRNA-seq)
can support association and *retrospective* discrimination, but they cannot
on their own establish a temporal "before". This charter records that limit
explicitly so downstream evaluation agents can downgrade or refuse claims
that exceed the data's epistemic warrant.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Claim:
    """A single testable scientific claim with explicit falsifiers.

    Attributes:
        name: Short identifier (e.g. ``"state"``, ``"when"``, ``"pathway"``).
        operational_meaning: A precise, instrument-level statement of what the
            claim asserts (no metaphor, no marketing).
        falsifiers: Concrete observations or test results that would refute
            the claim. Each falsifier should be checkable by an evaluation
            agent (Tier A--C).
        epistemic_limits: Caveats about the data, study design, or population
            that bound the claim's reach. Especially important for any
            temporal ("before it happens") language.
        required_evidence: The artefacts an evaluation agent must inspect to
            decide whether the claim is supported or falsified.
    """

    name: str
    operational_meaning: str
    falsifiers: list[str] = field(default_factory=list)
    epistemic_limits: list[str] = field(default_factory=list)
    required_evidence: list[str] = field(default_factory=list)


@dataclass
class Charter:
    """Top-level evaluation charter.

    Attributes:
        claims: The set of testable claims the pipeline is allowed to make.
        non_goals: Things ResistanceMap explicitly does *not* claim. These
            protect against scope creep during review.
        stop_rules: Conditions under which the evaluation layer must hard-stop
            (e.g. Tier A FAIL, charter violation, missing audit trail).
        notes: Free-form epistemic notes carried alongside the charter.
    """

    claims: list[Claim] = field(default_factory=list)
    non_goals: list[str] = field(default_factory=list)
    stop_rules: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def get_claim(self, name: str) -> Claim:
        """Look up a claim by name.

        Args:
            name: Claim identifier.

        Returns:
            The matching :class:`Claim`.

        Raises:
            KeyError: If no claim with that name is registered.
        """
        for claim in self.claims:
            if claim.name == name:
                return claim
        raise KeyError(f"No claim named {name!r} in charter")


def load_default_charter() -> Charter:
    """Build the default multiple-myeloma resistance charter.

    Three claims are encoded:

    * **state** -- the pipeline emits a discrete or continuous label for a
      patient's resistance phenotype against a named drug or drug class.
    * **when** -- the pipeline emits a time-to-event distribution or a
      horizon-bucket prediction over ``forecast_horizons = [3, 6, 12]``
      months as defined in :class:`~resistancemap.config.TrajectoryConfig`.
    * **pathway** -- the pipeline emits attributions on the PPI graph that
      identify the *mechanistic* drivers of the predicted resistance state.

    Returns:
        A fully populated :class:`Charter`.
    """
    state_claim = Claim(
        name="state",
        operational_meaning=(
            "Given a patient's multi-omic profile, the model emits a "
            "resistance phenotype label (discrete class or calibrated "
            "continuous score in [0, 1]) for each drug in "
            "config.data.target_drugs. The label is meant to reflect the "
            "patient's *current* resistance state at the time the omics "
            "snapshot was taken, not a future state."
        ),
        falsifiers=[
            "Brier score / ECE under patient-level (not sample-level) "
            "cross-validation is no better than a class-prevalence baseline.",
            "Discrimination (AUROC / AUPRC) collapses when train and test "
            "are split by patient ID instead of by sample.",
            "Calibration drift across cohorts (CCLE -> MMRF -> scRNA) "
            "exceeds documented thresholds with no recalibration plan.",
            "Label definition leaks the outcome (e.g. resistance label "
            "derived from a feature that is itself in the input).",
        ],
        epistemic_limits=[
            "Cross-sectional cell-line data (CCLE / GDSC) cannot distinguish "
            "primary from acquired resistance without explicit longitudinal "
            "follow-up.",
            "Single-cell snapshots support state inference only at the time "
            "of sampling; they do not by themselves license temporal claims.",
        ],
        required_evidence=[
            "Patient-level CV split manifest with no ID overlap.",
            "Reliability diagram and ECE per drug in target_drugs.",
            "Resistance label provenance document: source feature, "
            "thresholding rule, primary vs secondary distinction.",
        ],
    )

    when_claim = Claim(
        name="when",
        operational_meaning=(
            "Given a pre-treatment or early on-treatment profile, the model "
            "emits either a time-to-event distribution (e.g. time to "
            "progression) or a horizon-bucket probability over the horizons "
            "configured in TrajectoryConfig.forecast_horizons (months: "
            "[3, 6, 12]). 'Before it happens' is only meaningful when the "
            "input strictly precedes the event in wall-clock time."
        ),
        falsifiers=[
            "Concordance index (Harrell's C or Uno's C) is not better than "
            "a Kaplan-Meier or last-observation-carried-forward baseline.",
            "Pinball loss / CRPS at horizons 3, 6, 12 months is no better "
            "than a constant-hazard baseline.",
            "No timestamp field on the training samples, so train-time can "
            "exceed test-time for the same patient (temporal leakage).",
            "Horizon-3 performance is reported on patients whose followup "
            "is shorter than 3 months (right-censored without proper "
            "handling).",
        ],
        epistemic_limits=[
            "True 'before it happens' validation requires prospective or "
            "strictly time-ordered retrospective data with reliable event "
            "timestamps. CCLE / GDSC cannot satisfy this.",
            "MMRF CoMMpass provides longitudinal clinical data and is the "
            "only currently configured source able to license a temporal "
            "claim, and only for the patients with sufficient follow-up.",
        ],
        required_evidence=[
            "Per-sample collection_date and event_date fields.",
            "Concordance / pinball / CRPS at each horizon vs baselines.",
            "Censoring distribution and minimum-follow-up policy.",
        ],
    )

    pathway_claim = Claim(
        name="pathway",
        operational_meaning=(
            "For each predicted resistance state, the model emits an "
            "attribution over the PPI graph (STRING-derived) that highlights "
            "the proteins / edges driving the prediction. The attribution is "
            "claimed to be *mechanistic*, not merely correlative, only when "
            "it is stable under perturbation and generalises to held-out "
            "pathways."
        ),
        falsifiers=[
            "Attributions are not stable under input perturbation (e.g. "
            "rank correlation < 0.5 across noise replicates).",
            "Holding out a pathway at training time and testing on it "
            "yields no better than random attribution recovery.",
            "Top-k attributed proteins are dominated by hubs / highly "
            "connected nodes regardless of input (graph-prior leakage).",
            "PPI version / coverage of config.data.target_drugs's surface "
            "targets (e.g. CD38 for Daratumumab, BCMA for talquetamab-class "
            "agents) is undocumented or below a stated threshold.",
        ],
        epistemic_limits=[
            "STRING confidence cutoff (config.data.ppi_confidence) bounds "
            "the graph that any attribution can possibly recover.",
            "Attribution methods on GNNs are known to be sensitive to "
            "architecture choice; cross-method agreement should be reported.",
        ],
        required_evidence=[
            "PPI graph version, edge count, and coverage of "
            "config.data.target_drugs surface targets.",
            "Perturbation-stability report for the chosen attribution "
            "method.",
            "Pathway-level holdout protocol and result.",
        ],
    )

    charter = Charter(
        claims=[state_claim, when_claim, pathway_claim],
        non_goals=[
            "ResistanceMap is not a diagnostic device and is not claimed to "
            "replace clinical judgement.",
            "ResistanceMap does not claim causal inference at the patient "
            "level; pathway attributions are mechanistic hypotheses, not "
            "verified causal effects.",
            "ResistanceMap does not claim generalisation to indications "
            "outside multiple myeloma without explicit revalidation.",
        ],
        stop_rules=[
            "Any Tier A agent returning a FAIL verdict triggers a hard-stop: "
            "Tier B / C / D agents are marked BLOCKED with the Tier A "
            "failure as the blocking reason.",
            "Missing or unreadable audit trail (verification chain) is a "
            "hard-stop regardless of tier.",
            "Any claim made by the pipeline that is not in this charter "
            "must be either added to the charter or removed from the "
            "pipeline before evaluation can pass.",
        ],
        notes=[
            "EPISTEMIC NOTE: 'Predict resistance before it happens' is only "
            "demonstrable with prospective or strictly time-ordered data. "
            "Cross-sectional snapshots (CCLE, GDSC, single-cell at one "
            "time point) bound this claim to *retrospective association*. "
            "The 'when' claim therefore restricts itself to data sources "
            "with reliable event timestamps and minimum-followup policies.",
        ],
    )

    logger.debug(
        "Loaded default charter: %d claims, %d non-goals, %d stop rules",
        len(charter.claims),
        len(charter.non_goals),
        len(charter.stop_rules),
    )
    return charter
