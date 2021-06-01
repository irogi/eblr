# Example

An example notebook has been provided to run the code. The same code has been included below with an associated graph.

```python
from eblr import EBLR
import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('data.csv', index_col=0)

X = df[['dow','promo','date']].to_numpy()
y = df[['sales']].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=14)

eblr = EBLR()
eblr.fit(X_train, y_train)

y_pred = eblr.predict(X_test)
y_intervals = eblr.predict_intervals(X_test)

fig, ax = plt.subplots(1)
line = np.linspace(0, len(y_test)-1, len(y_test)).reshape(-1)

ax.plot(y_pred, color='g', label='Predicted Data')
ax.plot(y_test, color='r', label='Real value')

ax.fill_between(line,
                y_intervals[0],
                y_intervals[4],
                color='g',
                alpha=.2,
                label='90% quantile')
ax.fill_between(line,
                y_intervals[1],
                y_intervals[3],
                color='g',
                alpha=.3,
                label='50% quantile')

ax.set_xlabel('Dates')
ax.set_ylabel('Forecast')
ax.set_title('Sample Forecast')

plt.legend(loc='upper right')
```

![sample_forecast](./sample_forecast.png)
