# kfold_cv_utils

Función genérica de validación cruzada K-Fold para cualquier DataFrame,
con soporte para regresión o clasificación.

```python
from kfold_cv import kfold_cv
from sklearn.linear_model import LinearRegression

scores = kfold_cv(df,
                  features=['LSTAT'],
                  target='MEDV',
                  model=LinearRegression(),
                  k=4,
                  shuffle=True,
                  random_state=103)
