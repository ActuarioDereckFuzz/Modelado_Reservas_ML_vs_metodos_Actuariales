from src.analysis import build_occurrence_view


def test_occurrence_view_has_one_row_per_period(
    simulations,
):

    for df in simulations.values():

        occurrence = build_occurrence_view(df)

        assert len(occurrence) == (
            df["accident_period"].nunique()
        )


def test_occurrence_view_has_132_months(
    simulations,
):

    for df in simulations.values():

        occurrence = build_occurrence_view(df)

        assert len(occurrence) == 132