from dataclasses import dataclass

import numpy as np

from .loan import Loan


@dataclass
class Scenario:
    extra: float = 0.0
    inv_return: float = 0.0
    pct_principal: float = 1.0
    loan: Loan
    monthly_returns: np.ndarray[float]

    def invested_balance(self) -> float:
        balance = self.loan.loan_amt
        invested_balance = 0.0
        for _i in range(self.loan.term):
            interest = balance * self.loan.rate / 12
            principal = self.loan.payment - interest
            principal += self.extra * self.pct_principal
            principal = min(principal, balance)
            invest = self.loan.payment + self.extra - principal - interest
            r = self.monthly_returns[_i].item()
            invested_balance = (invested_balance + invest) * (1 + r)
            balance -= principal
        return invested_balance
