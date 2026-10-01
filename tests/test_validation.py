from src.validation import validate_all_scenarios


EXPECTED_DEV_M = {
    "creciente": 60,
    "decreciente": 48,
    "mixto": 48,
}


def test_all_simulated_bases_pass_structural_validation(
    simulations,
):
    results = validate_all_scenarios(
        simulations=simulations,
        expected_dev_M=EXPECTED_DEV_M,
        expected_start="2015-01",
        expected_end="2025-12",
    )

    assert results["passed"].all()