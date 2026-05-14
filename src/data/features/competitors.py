import pandas as pd

def process_competitor_variables(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Process competitor variables by replacing missing values, creating price comparison flags,
    and aggregating the total competitor landscape into compact summary features.
    """
    # Track the newly created columns to sum them up later
    better_cols = []
    worse_cols = []
    missing_cols = []

    for i in range(1, 9):
        rate_col = f'comp{i}_rate'
        inv_col = f'comp{i}_inv'
        diff_col = f'comp{i}_rate_percent_diff'

        missing_col = f'comp{i}_missing'
        has_better_price_col = f'comp{i}_has_better_price'
        has_worse_price_col = f'comp{i}_has_worse_price'

        # 1. Create Missingness and Price Flags
        df[missing_col] = (df[rate_col].isna() | df[inv_col].isna() | df[diff_col].isna()).astype('uint8')
        df[has_better_price_col] = (df[rate_col] < 0.0).fillna(False).astype('uint8')
        df[has_worse_price_col] = (df[rate_col] > 0.0).fillna(False).astype('uint8')

        # 2. Append to our tracking lists for the summary features
        missing_cols.append(missing_col)
        better_cols.append(has_better_price_col)
        worse_cols.append(has_worse_price_col)

        # 3. Impute missing values
        df[inv_col] = df[inv_col].where(df[missing_col] == 0)
        df[diff_col] = df[diff_col].where(df[missing_col] == 0)

        # 4. Drop the original rate column as it is now perfectly encoded in the flags
        if rate_col in df.columns:
            df.drop(columns=[rate_col], inplace=True)

    # 5. Generate Summary Aggregate Features (Horizontal Summation)
    # Because we tracked the exact columns in the loop, we don't need 'if' checks here
    df['comp_better_price_count'] = df[better_cols].sum(axis=1).astype('uint8')
    df['comp_worse_price_count'] = df[worse_cols].sum(axis=1).astype('uint8')
    df['comp_missing_count'] = df[missing_cols].sum(axis=1).astype('uint8')

    return df