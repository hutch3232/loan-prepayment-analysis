from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize_scalar

from .loan import Loan
from .returns import Returns


@dataclass
class AnalysisResult:
    """Holds the results of a loan analysis run."""

    xs: np.ndarray
    ys: list[float]
    opt_pct_principals: np.ndarray
    opt_means: np.ndarray
    opt_stds: np.ndarray
    risk_aversions: np.ndarray


@dataclass
class LoanAnalysis:
    loan: Loan
    returns: Returns
    monthly_returns: np.ndarray | None = None

    def generate_returns(self, n: int, seed: int | None = None):
        """Generate and store monthly returns."""
        self.monthly_returns = self.returns.generate_returns(
            n=n,
            term=self.loan.term,
            seed=seed,
        )

    def print_amortization(
        self,
        extra: float = 0.0,
        inv_return: float = 0.0,
        pct_principal: float = 1.0,
    ):
        self.loan.print_amortization(
            extra=extra,
            inv_return=inv_return,
            pct_principal=pct_principal,
        )

    def invested_balances_for_pct_principal(
        self,
        pct_principal: float,
        extra: float = 0.0,
    ) -> np.ndarray:
        """Vectorized evaluation of final invested balances for each simulation."""
        if self.monthly_returns is None:
            raise ValueError(
                "monthly_returns not generated. Call generate_returns(n) first."
            )

        monthly = self.monthly_returns
        n_sims, term = monthly.shape

        balance = np.full(n_sims, self.loan.loan_amt, dtype=float)
        invested = np.zeros(n_sims, dtype=float)
        payment = self.loan.payment
        monthly_rate = self.loan.rate / 12.0

        for m in range(term):
            interest = balance * monthly_rate
            principal = payment - interest + extra * pct_principal
            principal = np.minimum(principal, balance)
            invest = payment + extra - principal - interest
            r = monthly[:, m]
            invested = (invested + invest) * (1 + r)
            balance = balance - principal

        return invested

    def stats_invested_balance_for_pct(
        self,
        pct_principal: float,
        extra: float,
    ) -> tuple[float, float]:
        """Calculates mean and variance of final invested balances."""
        balances = self.invested_balances_for_pct_principal(pct_principal, extra=extra)
        return balances.mean().item(), balances.var().item()

    def mean_invested_balance_for_pct(
        self,
        pct_principal: float,
        extra: float,
    ) -> float:
        return (
            self.invested_balances_for_pct_principal(pct_principal, extra=extra)
            .mean()
            .item()
        )

    def optimize_pct_principal(
        self,
        extra: float,
        risk_aversion: float = 0.0,
        bounds=(0.0, 1.0),
    ):
        """
        Find the pct_principal that maximizes a risk-adjusted mean invested balance.
        The objective function is mean - risk_aversion * variance / 2.
        """

        def objective_func(p: float) -> float:
            mean, variance = self.stats_invested_balance_for_pct(p, extra=extra)
            # 1/2 factor for variance for nicer derivatives
            return -(mean - risk_aversion * variance / 2)

        return minimize_scalar(
            fun=objective_func,
            bounds=bounds,
            method="bounded",
        )

    def run_analysis(
        self,
        extra: float,
        risk_aversions: list[float],
    ) -> AnalysisResult:
        """
        Runs the analysis to find the optimal pct_principal for various risk
        aversion levels, creating data for an efficient frontier plot.
        """
        # compute objective curve for plotting (mean balance vs pct principal)
        xs = np.linspace(0.0, 1.0, 101)
        ys = [self.mean_invested_balance_for_pct(x, extra=extra) for x in xs]

        opt_pct_principals = []
        opt_means = []
        opt_stds = []

        if not isinstance(risk_aversions, list):
            risk_aversions = [risk_aversions]

        for ra in risk_aversions:
            opt_x = self.optimize_pct_principal(extra=extra, risk_aversion=ra).x
            mean, var = self.stats_invested_balance_for_pct(opt_x, extra=extra)

            opt_pct_principals.append(opt_x)
            opt_means.append(mean)
            opt_stds.append(np.sqrt(var))

        return AnalysisResult(
            xs=xs,
            ys=ys,
            opt_pct_principals=np.array(opt_pct_principals),
            opt_means=np.array(opt_means),
            opt_stds=np.array(opt_stds),
            risk_aversions=np.array(risk_aversions),
        )

    def plot_balance_distribution(
        self,
        extra: float,
        quantiles: list[float] | None = None,
    ):
        """
        Plots quantiles of the final invested balance against pct_principal
        to visualize the distribution of outcomes.
        """
        if quantiles is None:
            quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]

        pcts = np.linspace(0.0, 1.0, 101)
        results_by_pct = np.array(
            [self.invested_balances_for_pct_principal(p, extra=extra) for p in pcts]
        )

        fig = go.Figure()

        for q in quantiles:
            quantile_line = np.quantile(results_by_pct, q=q, axis=1)
            name = f"{q * 100:.0f}th Percentile"
            if q == 0.5:
                name = "Median (50th Percentile)"
            fig.add_trace(go.Scatter(x=pcts, y=quantile_line, mode="lines", name=name))

        mean_line = np.mean(results_by_pct, axis=1)
        fig.add_trace(
            go.Scatter(
                x=pcts,
                y=mean_line,
                mode="lines",
                name="Mean",
                line={"dash": "dash", "color": "black"},
            )
        )

        fig.update_layout(
            title="Distribution of Final Invested Balance vs. Payment Strategy"
            f"<br><sup>Expected Return: {self.returns.expected_return:.2%}, "
            f"Volatility: {self.returns.volatility:.2%}</sup>",
            xaxis_title="Percent of Extra Payment to Principal",
            yaxis_title="Final Invested Balance ($)",
            legend_title="Statistic",
        )
        fig.show()

    def plot_results(self, results: AnalysisResult):
        """Plots the results of a loan analysis."""
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=results.xs,
                y=results.ys,
                mode="markers",
                name="mean invested balance",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=results.opt_pct_principals,
                y=results.opt_means,
                mode="markers",
                name="optimal points",
                marker={"color": "red"},
            )
        )
        fig.update_layout(
            xaxis_title="Percent of Extra Payment to Principal",
            yaxis_title="Mean Ultimate Invested Balance",
        )
        fig.show()

    def plot_efficient_frontier(self, results: AnalysisResult):
        """Plots the efficient frontier."""
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=results.opt_stds,
                y=results.opt_means,
                mode="markers+text",
                name="efficient frontier",
                text=[f"ra={ra}" for ra in results.risk_aversions],
                textposition="top center",
            )
        )
        fig.update_layout(
            xaxis_title="Standard Deviation of Invested Balance",
            yaxis_title="Mean Invested Balance",
        )
        fig.show()
