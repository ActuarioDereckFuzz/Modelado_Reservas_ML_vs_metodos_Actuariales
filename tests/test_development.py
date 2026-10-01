from src.analysis import (
    calculate_individual_age_to_age_factors,
    calculate_volume_weighted_factors,
)

from src.validation import (
    validate_development_patterns,
)


def _calculate_factors(simulations):

    individual = []

    for df in simulations.values():
        individual.append(
            calculate_individual_age_to_age_factors(df)
        )

    import pandas as pd

    individual = pd.concat(
        individual,
        ignore_index=True,
    )

    aggregated = calculate_volume_weighted_factors(
        individual
    )

    return individual, aggregated


def test_individual_factor_count(simulations):

    for df in simulations.values():

        factors = (
            calculate_individual_age_to_age_factors(df)
        )

        n_accident_periods = (
            df["accident_period"].nunique()
        )

        dev_M = int(df["dev_M"].iloc[0])

        assert len(factors) == (
            n_accident_periods * dev_M
        )


def test_volume_weighted_factor_count(simulations):

    for df in simulations.values():

        individual = (
            calculate_individual_age_to_age_factors(df)
        )

        aggregated = (
            calculate_volume_weighted_factors(
                individual
            )
        )

        dev_M = int(df["dev_M"].iloc[0])

        assert len(aggregated) == dev_M


def test_age_to_age_factors_are_positive(simulations):

    for df in simulations.values():

        factors = (
            calculate_individual_age_to_age_factors(df)
        )

        assert (
            factors["age_to_age_factor"] > 0
        ).all()


def test_expected_development_patterns(simulations):

    _, aggregated = _calculate_factors(simulations)

    results = validate_development_patterns(
        aggregated,
        min_share=0.80,
    )

    assert results["passed"].all()