from dataclasses import dataclass, field

from amortization.amount import calculate_amortization_amount


@dataclass
class Loan:
    loan_amt: float
    rate: float
    term: int
    payment: float = field(init=False)

    def __post_init__(self) -> None:
        self.payment = calculate_amortization_amount(
            self.loan_amt, self.rate, self.term
        )

    def amortization_schedule(
        self,
        extra: float = 0.0,
        inv_return: float = 0.0,
        pct_principal: float = 1.0,
    ) -> tuple[list[dict[str, float]], float]:
        if extra < 0:
            raise ValueError("Extra payment cannot be negative.")
        if not (0.0 <= pct_principal <= 1.0):
            raise ValueError("pct_principal must be between 0 and 1, inclusive.")

        schedule = []
        balance = self.loan_amt
        interest_paid = 0.0
        invested_balance = 0.0

        for i in range(1, self.term + 1):
            interest = balance * self.rate / 12
            interest_paid += interest
            principal = self.payment - interest
            principal += extra * pct_principal
            principal = min(principal, balance)
            balance -= principal
            invest = self.payment + extra - principal - interest
            invested_balance = (invested_balance + invest) * (1 + inv_return) ** (
                1 / 12
            )

            schedule.append(
                {
                    "Month": i,
                    "Payment": principal + interest,
                    "Principal": principal,
                    "Interest": interest,
                    "Balance": balance,
                    "Invested": invest,
                    "Invested Balance": invested_balance,
                }
            )

        return schedule, interest_paid

    def print_amortization(
        self,
        extra: float = 0.0,
        inv_return: float = 0.0,
        pct_principal: float = 1.0,
    ) -> None:
        schedule, total_interest_paid = self.amortization_schedule(
            extra, inv_return, pct_principal
        )
        for entry in schedule:
            print(
                f"Month {entry['Month']:3d}: "
                f"Payment: {entry['Payment']:10,.2f} | "
                f"Principal: {entry['Principal']:10,.2f} | "
                f"Interest: {entry['Interest']:10,.2f} | "
                f"Balance: {entry['Balance']:10,.2f} | "
                f"Invested: {entry['Invested']:10,.2f} | "
                f"Invested Balance: {entry['Invested Balance']:10,.2f}"
            )
        print(f"Total interest paid: {total_interest_paid:,.2f}")
