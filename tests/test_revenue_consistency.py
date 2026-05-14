import pytest
torch = pytest.importorskip("torch")
from src.losses import implied_cohort_counts, implied_cohort_revenue


def test_revenue_consistency_math():
    cohort_size = torch.tensor([10.0, 5.0])
    txns_pc = torch.tensor([0.2, 0.4])
    avg_payment = torch.tensor([100.0, 20.0])
    visits_pc = torch.tensor([1.5, 2.0])
    visits, txns = implied_cohort_counts(cohort_size, visits_pc, txns_pc)
    revenue = implied_cohort_revenue(cohort_size, txns_pc, avg_payment)
    assert torch.allclose(visits, torch.tensor([15.0, 10.0]))
    assert torch.allclose(txns, torch.tensor([2.0, 2.0]))
    assert torch.allclose(revenue, torch.tensor([200.0, 40.0]))
    assert revenue.sum().item() == 240.0
