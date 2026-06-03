"""Non-GNN baseline models (target-agnostic).

Baselines quantify how much structural information the GNN
captures beyond simpler approaches:
  1. Linear combination: weighted element average
  2. Composition + Random Forest
  3. Composition + XGBoost
  4. Composition + MLP (neural network)
  5. Composition + Decision Tree
  6. Composition + kNN
  7. SOAP + KRR: structural descriptors

All baselines use the same train/val/test split as the GNN.
``get_soap_config`` takes ``include_hydrogen`` as a parameter
because the choice depends on the target:
  - hads: H is part of the adsorbed structure, include it.
  - wf:   bare slabs only, do not include H.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)

SUPPORTED_ELEMENTS: list[str] = [
    "Ag", "Au", "Co", "Cu", "Ir", "Ni",
    "Os", "Pd", "Pt", "Re", "Rh", "Ru",
]
N_ELEMENTS: int = len(SUPPORTED_ELEMENTS)
ELEMENT_TO_INDEX: dict[str, int] = {
    elem: i for i, elem in enumerate(SUPPORTED_ELEMENTS)
}

BASELINE_NAMES: list[str] = [
    "linear", "rf", "xgboost", "lightgbm", "catboost",
    "mlp", "decision-tree", "knn", "soap-krr",
]

CHECKPOINT_SCHEMA_VERSION: int = 1

VALID_FEATURE_KINDS: tuple[str, ...] = ("composition", "soap")

SOAP_R_CUT: float = 6.0
SOAP_N_MAX: int = 6
SOAP_L_MAX: int = 4
SOAP_AVERAGE: str = "outer"
SOAP_PERIODIC: bool = True


def get_soap_config(
    elements: list[str], include_hydrogen: bool,
) -> dict[str, Any]:
    """Build the SOAP descriptor config for the given metal list.

    The returned dict is unpacked into ``dscribe.descriptors.SOAP``
    both during training (descriptor computation) and during
    inference (descriptor reconstruction from the checkpoint).

    Args:
        elements: Metal element list (H is appended automatically
            when ``include_hydrogen`` is True).
        include_hydrogen: Whether to include H in the SOAP species
            list. True for hads (adsorbed slabs), False for wf
            (bare slabs).

    Returns:
        Dict suitable for ``SOAP(**config)``.
    """
    assert len(elements) > 0, "elements must not be empty"
    species = list(elements)
    if include_hydrogen:
        species.append("H")
    return {
        "species": species,
        "r_cut": SOAP_R_CUT,
        "n_max": SOAP_N_MAX,
        "l_max": SOAP_L_MAX,
        "average": SOAP_AVERAGE,
        "periodic": SOAP_PERIODIC,
    }


def build_lincomb_model(
    pure_values: dict[str, float],
    elements: list[str],
) -> LinearRegression:
    """Build a fitted LinearRegression that encodes the lincomb rule.

    The returned model has ``coef_`` set to the per-element pure
    reference value and ``intercept_ = 0``, calling ``predict(X)`` on
    a composition feature matrix ``X`` of shape ``[n, len(elements)]``
    returns the composition-weighted sum.

    Elements missing from ``pure_values`` are filled with 0.0 and a
    warning is logged.

    Args:
        pure_values: Element symbol -> reference value (e.g. mean
            pure-metal adsorption energy or work function in eV).
        elements: Canonical element ordering for the input feature
            vector (must match the ordering used by
            ``composition_to_vector`` at training and inference).

    Returns:
        Pre-fitted ``LinearRegression`` instance.
    """
    assert len(elements) > 0, "elements must not be empty"
    missing = [e for e in elements if e not in pure_values]
    if missing:
        logger.warning(
            "Lincomb: no pure ref for %s, using 0.0", missing,
        )
    coef = np.array(
        [float(pure_values.get(e, 0.0)) for e in elements],
        dtype=float,
    )
    model = LinearRegression(fit_intercept=False)
    model.coef_ = coef
    model.intercept_ = 0.0
    model.n_features_in_ = len(elements)
    return model


def save_baseline_checkpoint(
    path: Path,
    model: Any,
    kind: str,
    feature_kind: str,
    feature_config: dict[str, Any],
    elements: list[str],
    dataset: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a baseline model with the self-describing envelope.

    The dict envelope lets inference reconstruct the featurizer
    without consulting training code.

    Args:
        path: Output ``.pt`` path.
        model: Trained sklearn estimator or ``Pipeline``.
        kind: Algorithm identifier (e.g. ``"linear"``, ``"soap-krr"``).
        feature_kind: One of ``VALID_FEATURE_KINDS``.
        feature_config: Params needed to reconstruct the featurizer
            (e.g. ``SOAP`` kwargs), ``{}`` for composition baselines.
        elements: Metal ordering used during training.
        dataset: Training dataset identifier (e.g. ``"fcc4-v1p1"``).
        metadata: Free-form extras (val/test MAE, n_train, etc.).

    Raises:
        ValueError: If ``feature_kind`` is not recognised.
    """
    if feature_kind not in VALID_FEATURE_KINDS:
        raise ValueError(
            f"Unknown feature_kind: {feature_kind}. "
            f"Valid: {VALID_FEATURE_KINDS}"
        )
    assert isinstance(path, Path), "path must be a Path"
    assert len(elements) > 0, "elements must not be empty"

    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": kind,
        "feature_kind": feature_kind,
        "feature_config": feature_config,
        "elements": list(elements),
        "model": model,
        "dataset": dataset,
        "metadata": metadata or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def composition_to_vector(
    composition: dict[str, float],
) -> np.ndarray:
    """Convert composition dict to fixed-length feature vector.

    Args:
        composition: Element symbol to mole fraction mapping

    Returns:
        Array of shape (N_ELEMENTS,) with fractional compositions
    """
    assert len(composition) > 0, "Composition must not be empty"
    vector = np.zeros(N_ELEMENTS, dtype=np.float64)
    for elem, frac in composition.items():
        if elem in ELEMENT_TO_INDEX:
            vector[ELEMENT_TO_INDEX[elem]] = frac
    return vector


def build_linear_model() -> Pipeline:
    """Build linear combination baseline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])


def build_rf_model() -> Pipeline:
    """Build random forest baseline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def build_xgboost_model() -> Pipeline:
    """Build XGBoost baseline.

    Raises:
        ImportError: If xgboost not installed
    """
    from xgboost import XGBRegressor

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def build_lightgbm_model() -> Pipeline:
    """Build LightGBM baseline.

    Raises:
        ImportError: If lightgbm not installed
    """
    from lightgbm import LGBMRegressor

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LGBMRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=63,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )),
    ])


def build_catboost_model() -> Pipeline:
    """Build CatBoost baseline.

    Raises:
        ImportError: If catboost not installed
    """
    from catboost import CatBoostRegressor

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", CatBoostRegressor(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            random_seed=42,
            verbose=0,
        )),
    ])


def build_mlp_model() -> Pipeline:
    """Build MLP (neural network) baseline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )),
    ])


def build_decision_tree_model() -> Pipeline:
    """Build decision tree baseline."""
    return Pipeline([
        ("model", DecisionTreeRegressor(
            max_depth=10,
            random_state=42,
        )),
    ])


def build_knn_model() -> Pipeline:
    """Build k-nearest neighbors baseline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(
            n_neighbors=10,
            weights="distance",
            n_jobs=-1,
        )),
    ])


def build_soap_krr_model() -> Pipeline:
    """Build SOAP + kernel ridge regression with CV tuning.

    Uses GridSearchCV over alpha and gamma to find optimal
    hyperparameters for the RBF kernel.

    Raises:
        ImportError: If sklearn/dscribe not available
    """
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "gamma": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    }

    krr = GridSearchCV(
        KernelRidge(kernel="rbf"),
        param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", krr),
    ])


def build_baseline(name: str) -> Pipeline:
    """Build a baseline model by name.

    Raises:
        ValueError: If unknown baseline name
    """
    assert name in BASELINE_NAMES, (
        f"Unknown baseline: {name}. "
        f"Valid: {BASELINE_NAMES}"
    )

    builders = {
        "linear": build_linear_model,
        "rf": build_rf_model,
        "xgboost": build_xgboost_model,
        "lightgbm": build_lightgbm_model,
        "catboost": build_catboost_model,
        "mlp": build_mlp_model,
        "decision-tree": build_decision_tree_model,
        "knn": build_knn_model,
        "soap-krr": build_soap_krr_model,
    }

    return builders[name]()
