from loan_opt.analysis import LoanAnalysis
from loan_opt.loan import Loan
from loan_opt.returns import Returns

mortgage = Loan(
    loan_amt=500_000,
    rate=0.05,
    term=360,
)

returns = Returns(
    expected_return=0.07,
    volatility=0.15,
    degrees_of_freedom=10,
)

returns.print_summary()

analysis = LoanAnalysis(mortgage, returns)
analysis.generate_returns(n=10_000, seed=42)
analysis.print_amortization()
analysis.print_amortization(extra=1000.0, pct_principal=0.1, inv_return=0.07)
analysis.invested_balances_for_pct_principal(pct_principal=0.5, extra=1000.0)

results = analysis.run_analysis(
    extra=1000.0,
    risk_aversions=[1e-7, 5e-6, 1e-6, 5e-5, 1e-5],
)
analysis.plot_balance_distribution(extra=1000.0)
analysis.plot_results(results)
analysis.plot_efficient_frontier(results)
