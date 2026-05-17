import pandas as pd

NON_FEATURE_COLUMNS = {
    # IDs / grouping columns
    'srch_id',
    'prop_id',
    # 'prop_country_id',
    # 'visitor_country_location_id',
    # 'site_id',

    # datetime columns should be converted to numeric features first
    'date_time',
    'checkin_date',
    'checkout_date',

    # target / leakage columns
    'relevance',
    'click_bool',
    'booking_bool',
    'gross_booking_usd',
    'gross_bookings_usd',
    'position',
}


def select_feature_columns(
    train_df: pd.DataFrame,
    min_unique_values: int = 2,
) -> list[str]:
    """
    Select usable numeric/bool feature columns.
    Fit this only on the training set.
    """
    candidate_cols = train_df.select_dtypes(include=['number', 'bool']).columns

    feature_cols = []
    for col in candidate_cols:
        if col in NON_FEATURE_COLUMNS:
            continue

        if train_df[col].nunique(dropna=False) < min_unique_values:
            continue

        feature_cols.append(col)

    return feature_cols