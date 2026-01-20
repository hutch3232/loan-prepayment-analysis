from dataclasses import dataclass
from pprint import pprint

import numpy as np
from scipy.stats import t


@dataclass
class Returns:
    expected_return: float
    volatility: float
    degrees_of_freedom: int

    def generate_returns(
        self,
        n: int,
        term: int,
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate monthly returns.

        Returns a NumPy array of shape (n, term) and also sets
        `self.monthly_returns` for backward compatibility.
        """
        monthly_mean_return = (1 + self.expected_return) ** (1 / 12) - 1
        monthly_vol = self.volatility / np.sqrt(12)
        df = self.degrees_of_freedom
        scale = monthly_vol * np.sqrt((df - 2) / df)
        return t.rvs(
            loc=monthly_mean_return,
            scale=scale,
            df=df,
            size=(n, term),
            random_state=seed,
        )

    def summary(self) -> dict:
        """Provides a summary of the statistical characteristics of the returns."""
        df = self.degrees_of_freedom
        scale = self.volatility * np.sqrt((df - 2) / df)

        dist = t(df=df, loc=self.expected_return, scale=scale)

        quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

        return {
            "mean": dist.mean().round(3).item(),
            "volatility": dist.std().round(3).item(),
            "excess-kurtosis": dist.stats(moments="k").round(3).item(),
            "skewness": dist.stats(moments="s").round(3).item(),
            "quantiles": {q: dist.ppf(q).round(3).item() for q in quantiles},
        }

    def print_summary(self) -> None:
        pprint(self.summary())
