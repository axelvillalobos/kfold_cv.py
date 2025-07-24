# ── imports ──────────────────────────────────────────────────────────────
import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import (
    r2_score,            # regresión
    mean_absolute_error, # regresión
    accuracy_score,      # clasificación
    f1_score             # clasificación binaria/multiclase
)

# ── función genérica ─────────────────────────────────────────────────────
def kfold_cv(
    df,
    features,             # lista o tupla de nombres de columnas predictoras
    target,               # nombre de la columna objetivo
    model,                # instancia de scikit-learn (o pipeline)
    k=5,                  # nº de folds
    shuffle=True,         # barajar o no
    random_state=None,    # para reproducibilidad si shuffle=True
    scoring=None,         # función métrica; si None se elige automáticamente
    return_scores=True,   # devolver la lista de scores además de imprimir
    verbose=True          # mostrar progreso por pantalla
):
    """
    Ejecuta K-Fold CV sobre df con las columnas indicadas y devuelve las
    puntuaciones por fold (y su media/desviación estándar).

    Ejemplo de uso:
    >>> scores = kfold_cv(df, ['LSTAT'], 'MEDV',
    ...                   LinearRegression(), k=4, random_state=103)
    """
    # ---- prepara datos --------------------------------------------------
    X = df[features].values
    y = df[target].values

    # ---- selecciona métrica si no la pasa el usuario --------------------
    if scoring is None:
        # Si el objetivo tiene muchos valores distintos asumimos regresión
        scoring = r2_score if len(np.unique(y)) > 20 else accuracy_score

    kfold = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)

    # ---- iterar sobre los folds -----------------------------------------
    fold_scores = []
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X), start=1):
        # clona el modelo para no sobre-escribirlo en cada fold
        m = clone(model)
        m.fit(X[train_idx], y[train_idx])

        y_pred = m.predict(X[test_idx])
        score  = scoring(y[test_idx], y_pred)
        fold_scores.append(score)

        if verbose:
            print(f"Fold {fold:>2}: {scoring.__name__} = {score:.4f}")

    # ---- resumen --------------------------------------------------------
    mean_score = np.mean(fold_scores)
    std_score  = np.std(fold_scores)
    if verbose:
        print(f"\n{scoring.__name__} medio: {mean_score:.4f} ± {std_score:.4f}")

    return fold_scores if return_scores else None
