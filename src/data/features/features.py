import pandas as pd
import numpy as np

def compute_comp_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the competition rate for each search.
    """
    for num in range(1, 9):
        percent_diff = pd.to_numeric(df[f'comp{num}_rate_percent_diff'], errors='coerce')
        price_usd = pd.to_numeric(df['price_usd'], errors='coerce')
        df[f'comp{num}_rate'] = percent_diff * price_usd

    return df

def build_relevance_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps clicks and bookings to the Kaggle competition relevance scale.
    Assumes your training data has 'click_bool' and 'booking_bool' columns.
    This is for building the target variable during training, not for test set predictions.
    """
    conditions = [
        df['booking_bool'] == 1,
        df['click_bool'] == 1
    ]
    choices = [5, 1]
    
    # 5 if booked, 1 if clicked (but not booked), 0 otherwise
    df['relevance'] = np.select(conditions, choices, default=0)

    return df

def build_binary_season(name: str, df: pd.DataFrame, start: int, end: int, ref: str = 'date_time') -> pd.DataFrame:
    """Build a boolean variable which flags whether a date-time of a query is within a specified
    period. For example, if one can define a high-season, one can check whether the query 
    happended within that season. The start and end boundaries are inclusive. 

    Args:
        name (string):      The name of the new variable
        df (pd.DataFrame):  The dataframe to edit
        start (int):        The starting month of the specified period
        end (int):          The last month of the specified period
        ref (string):       The variable to reference. For example query datetime, 
                            or check-in date. 
    """
    df[name] = df[ref].dt.month.between(start, end) # .between is inclusuve !
    return df

def build_checkin_dates(df: pd.DataFrame) -> pd.DataFrame:
    """ Builds a variable which determines the intended check-in date in a search query 
    from 'date_time' and 'srch_booking_window'. 

    Args:
        df (pd.DataFrame)   : The dataframe to edit
    """
    df['checkin_date'] = df['date_time'] + pd.to_timedelta(df['srch_booking_window'].squeeze(), unit='D') # type: ignore[arg-type]
    return df

def build_flag_variable(name: str, df: pd.DataFrame, variable: str, value: float = 0.0) -> pd.DataFrame:
    """ Replaces specified values in variable with NA, and creates a new variable that 
    denotes when NA has occured for a hotel.

    Args: 
        df (pd.DataFrame)   : The dataframe to edit
        variable (string)   : The variable to check for specified value
        value (float)       : The value to replace with pd.NA
    """
    # First, create flag variable
    df[name] = df[variable][df[variable] == value].astype(int)
    # Then, replace the specified values with NAs
    df[variable] = df[variable].replace(value, np.float64('nan'))
    return df

def build_persona_one_hot_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the parent and children persona variables
    """
    df['srch_adults_count'] = pd.Categorical(df['srch_adults_count'], categories=range(6))
    df['srch_children_count'] = pd.Categorical(df['srch_children_count'], categories=range(5))

    return pd.get_dummies(
        df,
        columns=['srch_adults_count', 'srch_children_count'],
        prefix=['adults', 'children'],
        dtype=int,
    )

def build_prop_avg_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the average price per property and adds it as a new column.
    """
    df['prop_avg_price'] = df.groupby('prop_id')['price_usd'].transform('mean')
    return df

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract model-usable date features from date_time and checkin_date
    """
    df['search_month'] = df['date_time'].dt.month.astype('uint8') # Months are 1-12, so uint8 is sufficient
    df['search_dayofweek'] = df['date_time'].dt.dayofweek.astype('uint8') # Monday=0, Sunday=6
    df['search_hour'] = df['date_time'].dt.hour.astype('uint8') # 0-23, so uint8 is sufficient

    if 'checkin_date' in df.columns:
        df['checkin_month'] = df['checkin_date'].dt.month.astype('uint8') 
        df['checkin_dayofweek'] = df['checkin_date'].dt.dayofweek.astype('uint8')
        df['checkin_weekend'] = df['checkin_dayofweek'].isin([5, 6]).astype('uint8')

    return df


def add_travel_party_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features describing the type of traveller/search party
     - Total guests
     - Whether the search includes children
     - Whether it's a family (2+ adults with children)
     - Whether it's a solo traveler (1 adult, no children)
     - Guests per room
    """
    df['srch_total_guests'] = df['srch_adults_count'] + df['srch_children_count'] # Assuming infants are not counted in the dataset, otherwise we would need to add them here as well
    df['srch_has_children'] = (df['srch_children_count'] > 0).astype('uint8') # Whether the search includes any children
    df['srch_is_family'] = (
        (df['srch_adults_count'] >= 2) & (df['srch_children_count'] > 0)
    ).astype('uint8') # Whether it's a family search (2 or more adults with children)
    df['srch_is_solo'] = (
        (df['srch_adults_count'] == 1) & (df['srch_children_count'] == 0)
    ).astype('uint8') # Whether it's a solo traveler search (1 adult, no children)

    rooms = df['srch_room_count'].clip(lower=1) # Avoid division by zero, treat 0 rooms as 1 room for the purpose of this feature
    df['srch_guests_per_room'] = df['srch_total_guests'] / rooms # Average number of guests per room, which can indicate whether they prefer shared rooms or private rooms

    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple price transformations
    """
    df['price_usd_log'] = np.log1p(df['price_usd']) # Log-transform price to reduce skew, add 1 to avoid log(0)

    nights = df['srch_length_of_stay'].clip(lower=1) # Avoid division by zero, treat 0 nights as 1 night for the purpose of these features
    guests = df['srch_total_guests'].clip(lower=1) if 'srch_total_guests' in df.columns else 1 # Use total guests if available, otherwise default to 1 to avoid division by zero

    # Price per night, price per guest, and price per guest-night can help the model learn about the value proposition of the hotel for different types of travelers and trip lengths
    df['price_per_night'] = df['price_usd'] / nights 
    df['price_per_guest'] = df['price_usd'] / guests
    df['price_per_guest_night'] = df['price_usd'] / (guests * nights)

    return df


def add_history_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare current hotel to user's historical preferences.
    Assumes visitor_hist_starrating and prop_starrating are already scaled similarly.
    """
    # For example, if a user historically prefers 4-star hotels, then a 4-star hotel in the current search might be more relevant than a 2-star or 5-star hotel. This feature captures that preference alignment.
    df['star_vs_user_history'] = (
        df['prop_starrating'].replace(-1, np.nan)
        - df['visitor_hist_starrating'].replace(-1, np.nan)
    ).fillna(0)

    # Similarly, if a user historically has an average daily rate (ADR) of $100, 
    # then a hotel priced around $100 might be more relevant than one priced at $50 or $200
    # This feature captures how the current hotel's price compares to the user's historical spending habits
    price_dollars = df['price_usd'] / 100
    hist_adr = df['visitor_hist_adr_usd'].replace(-1, np.nan)

    df['price_vs_user_hist_adr'] = (
        (price_dollars - hist_adr) / hist_adr.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


def add_competitor_summary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate competitor information into compact features
        - Count of competitors with better price
        - Count of competitors with worse price
        - Count of missing competitor information
    """

    # The dataset has up to 8 competitors, and for each competitor, 
    # there are features indicating whether that competitor has a better price, 
    # a worse price, or if the information is missing
    # Instead of having 24 separate binary features (3 for each of the 8 competitors)
    # We can summarize this information into just 3 features that count how many competitors fall into each category
    # This reduces dimensionality and can help the model focus on the overall competitive landscape rather than individual competitors.
    better_cols = [f'comp{i}_has_better_price' for i in range(1, 9) if f'comp{i}_has_better_price' in df.columns]
    worse_cols = [f'comp{i}_has_worse_price' for i in range(1, 9) if f'comp{i}_has_worse_price' in df.columns]
    missing_cols = [f'comp{i}_missing' for i in range(1, 9) if f'comp{i}_missing' in df.columns]

    if better_cols:
        df['comp_better_price_count'] = df[better_cols].sum(axis=1).astype('uint8')
    if worse_cols:
        df['comp_worse_price_count'] = df[worse_cols].sum(axis=1).astype('uint8')
    if missing_cols:
        df['comp_missing_count'] = df[missing_cols].sum(axis=1).astype('uint8')

    return df


def add_query_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add within-search features.
    These are very important for learning-to-rank because hotels compete within srch_id.
    """
    cols = [
        'price_usd',
        'price_usd_log',
        'price_per_night',
        'prop_starrating',
        'prop_review_score',
        'prop_location_score1',
        'prop_location_score2',
        'prop_log_historical_price',
        'srch_query_affinity_score',
        'orig_destination_distance',
    ]

    # For each of these columns, we will compute features that describe 
    # how the hotel's value compares to other hotels in the same search (srch_id)
    new_cols = {}
    for col in cols:
        if col not in df.columns:
            continue

        x = df[col].replace(-1, np.nan)
        g = x.groupby(df['srch_id'])

        mean = g.transform('mean')
        min_val = g.transform('min')
        max_val = g.transform('max')

        new_cols[f'{col}_minus_query_mean'] = (x - mean).fillna(0)
        new_cols[f'{col}_div_query_mean'] = (
            x / mean.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(1)

        new_cols[f'{col}_query_pct_rank'] = g.rank(pct=True).fillna(0)
        new_cols[f'{col}_is_query_min'] = (x == min_val).fillna(False).astype('uint8')
        new_cols[f'{col}_is_query_max'] = (x == max_val).fillna(False).astype('uint8')

    if new_cols:
        new_df = pd.DataFrame(new_cols, index=df.index)
        df = pd.concat([df, new_df], axis = 1)
    return df  
