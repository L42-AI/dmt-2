#!/usr/bin/env python
# coding: utf-8

# # EDA
# 

# In[1]:


# System Libraries
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

# Added libraries and modules
from diagnostics import diagnose_data
import pandas as pd
import matplotlib.pyplot as plt
from data import load_data
pd.set_option('display.max_columns', 54)


# In[45]:


# Load data
train_set, test_set = load_data()


# In[52]:


train_set[train_set['random_bool'] == 1].groupby('srch_id').size().min()


# ## Data shape and table
# 

# In[3]:


n_rows = 5
display(train_set.head(n_rows))
display(test_set.head(n_rows))
print('train_set shape:', train_set.shape)
print('test_set shape:', test_set.shape)


# In[4]:


display("All attributes in training set: ", train_set.columns)
train_excl_attr = set(train_set.columns) - set(test_set.columns)
display("training exclusive attributes: ", train_excl_attr)


# The above show the following:
# - Both the training and test sets have close to 5 million instances.
# - Training set has 54 attributes, test set has 50
# - The training exclusive attributes are:
#     - Booking_bool, click_bool, gross_bookings_usd, position

# ## Evaluation Goal
# The evaluation is given in the assignment document under "Winning the competition". The evaluation metric is NDCG@5 (Normalized Discounted Cumulative Gain). The @5 designates that they will look at ranks 1-5 in each search query. 
# 
# Each query contains a set of hotels. The hotels are assigned following values:
# - 5 = the user purchased a room
# - 1 = the user clicked through to see more information on this hotel
# - 0 = the user did neither
# 
# For each hotel in the training set, this value can be derived from "booking_bool" and "click_bool"
# 
# The NDCG@5 takes the top 5 ranked hotels in our prediction and computes the sum of their "relevance scores" (the values above). It then applies penalization for highly relevant hotels having lower ranks (e.g. if the hotel booked is ranked 5th), and normalizes the score against varying number of hotel listings per search. 
# The average across all queries in the test set is taken to determine the performance of our model in the competition. 

# ##### Questions/Notes:
# - The evaluation metric they use is on the query-level and not the hotel-level. So each query could be seen as a training instance itself. So we could engineer non-parametric transformations into ranks or z-standardizations of the original features which makes the listings more comparable across queries. For example, a query with big budget vs one with low budget, the absolute price of the hotels doesn't matter to the user, only the price relative to all other hotels. 
# - While this is the competition evaluation, we can train models using different evaluation (hotels that were booked or clicked have value 1, then it can be a logistic regression model). There is also a model which uses this specific evaluation metric as loss: LambdaMART (https://www.shaped.ai/blog/lambdamart-explained-the-workhorse-of-learning-to-rank) 
#     - One team who participated in this competition used the following models (https://arxiv.org/pdf/1311.7679): 
#         - LIBLINEAR and SVMRankfor pairwise logistic regression; 
#         - Random Forest from scikit-learn 
#         - Ranking algorithms like AdaRank, LambdaMART from RankLib2. 
#         - Gradient Boosting Machine3, Extremely Randomized Trees4 from R, 
#         - deep neural network implementation from PyLearn2
#         - Factorization Machine libFM5
#     - LamgdaMART is computationally intensive to train, relies on good feature engineering, and is sensitive to hyperparameters D:)
# 

# ## Sparsity

# In[5]:


sparsity = sum((train_set == 0).astype(int).sum())/train_set.size
print('sparsity of train_set:', sparsity)
diagnose_data(train_set, name = 'train_set')
display(train_set.isna().sum())


# - The training set is not particularly sparse
# - However, close to half of the data has missing values, which is a problem.
# - It's likely that this is systematic missingness: Looking at the first 5 entries of both sets, missingness seems to be pervasive within specific variables:
#     - "visitor_hist_starrating", "visitor_hist_adr_usd" : Nulls mean that there is no user history for purchases -> Many customers are new customers. Imputation here could be a neutral value, for example neutral rating and mean average purchase history.
#     - "srch_query_affinity_score" : Null means that hotel did not register in searches -> Many hotels did not register in search queries.
#     - "orig_destination_distance" : Null means that distance could not be calculated. -> This is potentially a useful attribute, some imputation heuristic could be useful here. For example, we could use "visitor_location_country_id" and compute the distance to the property. 
#     - "comp* attributes" : Null means there is no competetive data -> There is no competetive data of certain competitors for certain hotels listed. Question here is what it means that there is no data? Either data is missing at random (measurement error) or competitor x does not have the property in their catalogue, hence no competetive data. The definition of "availability" in the attribute description is not clear: Does no availability mean they have the property, but there are no rooms available, or they do not have the property?

# ## 

# The evaluation metric requires at least 5 listings per search. Also, the attributes "click_bool" and "booking_bool" ideally do not need imputation. To know if this holds:

# In[6]:


train_grouped = train_set.groupby('srch_id')
n_search = list() 

for srch_id, data in train_grouped:
    n_search.append(data.shape[0])


# In[7]:


plt.hist(n_search, bins = 100)
plt.title('Distribution of property counts per search')
plt.xlabel('Property Counts')
plt.ylabel('Frequency')
plt.show()


# In[8]:


train_set[['booking_bool', 'click_bool']].isna().sum()


# Thankfully all searches seem to have resulted in at least 5 properties, the evaluation metric is applicable. Thankfully the attributes that the evaluate metric depends on have no missing values, hence we do not need to impute those.
# 
# ## Number of queries
# 
# We have almost 200 000 searches in the training set, likewise in the test set. 

# In[9]:


query_count_train = train_set['srch_id'].unique().size
display('Number of search queries in the training set: ', query_count_train)


# ## Univariate Analysis

# ### Target variable: Relevance
# The evaluation target is the NDCG per query, which requires relevance scores per hotel in a query. We need to engineer a feature called "rel_score" from "click_bool" and "booking_bool":

# In[10]:


which_clicked = train_set.query('click_bool == 1').index
which_booked = train_set.query('booking_bool == 1').index
which_neither = train_set.query('click_bool == 0 & booking_bool == 0').index

train_set.loc[which_clicked, 'rel_score'] = 1
train_set.loc[which_booked, 'rel_score'] = 5
train_set.loc[which_neither, 'rel_score'] = 0

display(train_set[['booking_bool', 'click_bool', 'rel_score']].query('rel_score != 0.0').head())


# When looking at relevance scores within individual search queries, we find that there are some (or many) queries where the user just clicks on one listing and then immediately books it. There are also queries where the user clicked on some listings, but does not book anything. 
# 
# Furthermore, 
# 
# So there are queries for which the maximal sum of relevance scores is 5: If the one booked is in the top 5 in your model's ranking, then you get the best possible evaluation by NDCG

# In[11]:


display(train_set.loc[which_clicked].query('srch_id == 1')) # clicked one and booked
display(train_set.loc[which_clicked].query('srch_id == 4')) # clicked one and didn't book anything


# In[12]:


# Every query has 1 booked property!
display('number of queries where nothing was booked', train_set.groupby('srch_id').max().query('booking_bool == 0').shape[0])
display('number of queries where nothing was clicked',train_set.groupby('srch_id').max().query('click_bool == 0').shape[0])


# In[13]:


# Plot the number of clicked hotels for all queries
clicked_counts = train_set.loc[which_clicked].groupby('srch_id').size()
plt.hist(clicked_counts, bins = 20)
plt.xlabel('Number of clicked hotels in query (including the booked hotel)')
plt.ylabel('Count queries')
plt.show()
single_click_query_count = (clicked_counts == 1).sum()
print('number of queries with 1 click only: ', single_click_query_count)


# 
# - There are 61405 queries which did not result in any booking
# - There are no queries that did not result in a click
# - The number of queries where there was only one click **by far outweight queries with more than one click**
# 
# From all queries that had one single click, how many resulted in that clicked hotel being booked?

# In[14]:


train_set_no_datetime = train_set.drop(columns=['date_time'])
queries_one_click_no_book = train_set_no_datetime.groupby('srch_id').sum().query('click_bool == 1 & booking_bool == 0')
queries_one_click = train_set_no_datetime.groupby('srch_id').sum().query('click_bool == 1')

print('ratio of queries with one click to queries with more than one click: ',
      single_click_query_count / query_count_train)
print('ratio of queries with no booking within queries with only one click: ',
      queries_one_click_no_book.shape[0] / queries_one_click.shape[0])


# **Conclusion**: Around 93% of the queries in the training data are single-click queries. And there are around 70% of these where the single click resulted in that hotel being booked. This means that around 65% of total queries will have only one hotel which was consequently clicked on and booked.  
# 
# What consequences does this have for the modelling? The output below illustrates this: If there is only a single non-zero relevance value in the top-5 (either 1 or 5), then the NDCG penalizes this more than if there are other non-zero values. Also, if there is only one non-single value, then it doesn't matter to the NDCG if it is 1 or 5. 
# 
# Since single-click/book queries are 93% of the total queries, this means that our model should be focusing on learning to put relatively high relevance scores on top as much as possible, and the penalty for putting higher relevance scores in lower ranks should be very high for the model to learn to do so. We could adjust the relevance scores for training, or adjust the NDCG formula to do so. 

# In[15]:


from evaluate import ndcg_5

one_click_ranked_scores = [5, 0, 0, 0, 0]
five_click_ranked_scores = [5, 1, 1, 1, 1]
one_click_no_book_ranked_scores = [5, 0, 0, 0, 0]

print('Best possible scores')
print('one_click_booked: ', ndcg_5(one_click_ranked_scores))
print('five_clicks: ', ndcg_5(five_click_ranked_scores))
print('one_click_not_booked: ', ndcg_5(one_click_no_book_ranked_scores))
print('==========')

one_click_ranked_scores = [0, 5, 0, 0, 0]
five_click_ranked_scores = [1, 5, 1, 1, 1]
one_click_no_book_ranked_scores = [0, 1, 0, 0, 0]
print('Second best possible scores')
print('one_click_booked: ', ndcg_5(one_click_ranked_scores))
print('five_clicks: ', ndcg_5(five_click_ranked_scores))
print('one_click_not_booked: ', ndcg_5(one_click_no_book_ranked_scores))
print('==========')

one_click_ranked_scores = [0, 0, 5, 0, 0]
five_click_ranked_scores = [1, 1, 5, 1, 1]
one_click_no_book_ranked_scores = [0, 0, 5, 0, 0]
print('Third best possible scores')
print('one_click_booked: ', ndcg_5(one_click_ranked_scores))
print('five_clicks: ', ndcg_5(five_click_ranked_scores))
print('one_click_not_booked: ', ndcg_5(one_click_no_book_ranked_scores))
print('==========')


# ### Other variables
# #### Datetime
# The date_times of the query are distributed within a range of around half a year: From November 2012 to June 2013. The distribution of query datetimes seems to show an increase in searches as summer comes along. The number of bookings alse increase as summer approaches. 

# In[16]:


range_date_time = (train_set['date_time'].min(), train_set['date_time'].max())
print("earliest search date: ", range_date_time[0])
print("latest search date: ", range_date_time[1])

query_times = train_set.groupby('srch_id')['date_time'].first()
plt.hist(query_times, bins = 9)
plt.title('Search query frequency over months')
plt.ylabel('Query frequency')
plt.xlabel('Months')
plt.show()

booking_times = train_set.loc[which_booked].groupby('srch_id')['date_time'].first()
plt.hist(booking_times, bins = 9)
plt.title('Booking frequency over months')
plt.ylabel('Query frequency')
plt.xlabel('Months')
plt.show()


# We might be interested in the intended date when the hotel stay should start (this could also be a feature as otherwise the model must learn this from "date_time" and "srch_booking_window").

# In[17]:


train_set['checkin_date'] = train_set['date_time'] + pd.to_timedelta(train_set['srch_booking_window'], unit='D')
counts = train_set.groupby('srch_id')['checkin_date'].first().dt.month.value_counts().sort_index()
counts = counts.reindex(range(1, 13), fill_value=0)
counts.plot(kind='bar')
plt.title("Intended checkin dates for all queries")
plt.xticks(range(12), ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
plt.show()

counts = train_set.query('booking_bool == 1').groupby('srch_id')['checkin_date'].first().dt.month.value_counts().sort_index()
counts = counts.reindex(range(1, 13), fill_value=0)
counts.plot(kind='bar')
plt.title('Intended checkin dates for queries where hotel was booked')
plt.xticks(range(12), ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
plt.show()


# There is a clear trend: There are off-seasons where desired checkins are low (Aug-Nov) and high seasons (Dec-July) where intended checkins are high. The same applies for queries where a booking was confirmed. We might benifit from creating a categorical variable which encodes whether the intended check-in date occurs in a high season.  

# # Property ratings

# In[18]:


prop_starrating = train_set['prop_starrating'].value_counts().sort_index()
prop_starrating.plot(kind='bar')
plt.xlabel('Hotel Star Ratings')
plt.ylabel('Frequency')
plt.show()


# There is a small fraction of hotel listings which have 0 values, these need to be turned to nas, as they are missing due to various (possible unknown) reasons. From looking at the plot, the star ratings seem to be normally distributed. 

# In[19]:


prop_review_score = train_set['prop_review_score'].value_counts().sort_index()
prop_review_score.plot(kind="bar")
plt.show()


# In[20]:


from data.features import build_flag_variable
train_set = build_flag_variable('flag_no_previous_reviews', train_set, 'prop_review_score', 0)
train_set = build_flag_variable('flag_no_stars_available', train_set, 'prop_starrating', 0)
train_set = build_flag_variable('flag_no_historical_price', train_set, 'prop_log_historical_price', 0)


# The distribution of user reviews show that the majority of hotels listed have positive user reviews. Zero values indicate that there are no previous reviews present (different from NA here which indicates that the information is not available for some reason).

# ### Search Requirements

# In[21]:


queries = train_set.groupby('srch_id').first()

length_of_stay = queries['srch_length_of_stay']
booking_window = queries['srch_booking_window']
adults_count = queries['srch_adults_count']
children_count = queries['srch_children_count']
room_count = queries['srch_room_count']


# In[22]:


length_of_stay.value_counts().sort_index().plot(kind='bar')
plt.title('Length of stay')
plt.show()
booking_window.value_counts().sort_index().plot(kind='bar')
plt.title('Booking window until checkin-date')
plt.xticks([20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
plt.show()
booking_window.value_counts().sort_index().head(20).plot(kind='bar')
plt.show()
adults_count.value_counts().sort_index().plot(kind='bar')
plt.show()
children_count.value_counts().sort_index().plot(kind='bar')
plt.show()
room_count.value_counts().sort_index().plot(kind='bar')
plt.show()


# Before analysing, we note the following: Booking window has no missing values, despite some queries resulting in no booking. We therefore assume that the booking window refers to the number of days in the future the *intended* hotel stay starts. 
# The distribution of the search reqquirements show us the following:
# - The majority of searches are looking for short term stays (1-4 days). The most frequent duration searched for is 1 day
# - A substantial amount of users started searching for hotels shortly before the intended checking date. The most often occuring window was 1 day before the intended check-in
# - Most people intend to travel in pairs, followed by lone travelers. 
# - These people mostly do not bring children. 
# - They want to mostly book single rooms. 

# ### Geographic variables
# 

# In[23]:


orig_destination_distance = train_set['orig_destination_distance']
orig_destination_distance.plot.kde()
plt.title("Destination distance")
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.show()


# Hard to interpret without knowing the unit of the distance. We can probably assume that unit is either kilometres (earth's circumference is around 25000 miles, so miles doesn't make sense given the range of values we see here). We can see that many distances are below 1000 km (which is the distance between central Germany and central France). We could interpret this as most queries are probably looking for destinations within the same country they are located in, or otherwise very close. In other words, most are short-distance travelers.
# 

# In[24]:


orig_destination_distance.min()


# ## Multivariate analysis
# 

# - Relationship between Hotel rating/stars/luxury to clicks and bookings
# - Relationship between various attributes and whether hotel was single_click query
#     - We cannot use click and booking in the test, but we can see how variables interact and perhaps gain some useful insights. We want out model to recognize somehow whether a query will likely be a single-click query or not!
# - Relationship between bookings, clicks, and frequency of hotel appearance (CTR & CVR scores, mentioned in article)
# 

# In[25]:


def build_interest(df: pd.DataFrame) -> pd.DataFrame:
    df['interest'] = df['click_bool'] | df['booking_bool']
    return df

def build_group_size(df:pd.DataFrame) -> pd.DataFrame:
    df['group_size'] = df['srch_adults_count'] + df['srch_children_count']
    return df
train_set = build_interest(train_set)
train_set = build_group_size(train_set)


# ### Hotel Star Rating vs Interest
# 
# We want to analyse how the 'expensiveness' or 'fanciness' of the hotel impacts user interest in the hotel. The variable 'price_usd' is not really useful, since the price is not uniform across countries, and it's not clear whether it is per night or stay. However, we may use the hotel star-rating as a substitute: Often higher stars reflect services or amneties, such as spas, that hotels with lower budgets cannot afford. For historical price, it is not clear what the trading period is, or whether it varies depending on location. Nonetheless, we may look at its trend. 

# In[4]:


import matplotlib.cm as cm
import numpy as np

def ordinal_scatter(df: pd.DataFrame, x: str, y: str, control: str = None, 
                    scatter: bool = True, bin_x: bool = False, 
                    bin_y: bool = False, bin_control: bool = False, 
                    q: int = 5):
    df = df.copy()
    df = df.dropna(subset=[x, y])
    if bin_x:
        df[x] = pd.qcut(df[x], q = q).apply(lambda i: i.mid)
    if bin_y:
        df[y] = pd.qcut(df[y], q = q).apply(lambda i: i.mid)

    # both x and y MUST be ordinal variables
    if control is None:

        y_means_per_x_level = df.groupby(x)[y].mean()
        x_levels = y_means_per_x_level.index
        if scatter: 
            plt.scatter(x_levels, y_means_per_x_level)
        else:
            plt.plot(x_levels, y_means_per_x_level)
    else:
        df = df.dropna(subset=[control])
        if bin_control:
            df[control] = pd.qcut(df[control], q = q).apply(lambda i: i.mid)
        levels_control = sorted(df[control].unique())
        colors = cm.coolwarm(np.linspace(0, 1, len(levels_control)))
        for level, color in zip(levels_control, colors):
            y_means = df[df[control] == level].groupby(x)[y].mean()
            if scatter:
                plt.scatter(y_means.index, y_means, color=color, label=str(level))
            else:
                plt.plot(y_means.index, y_means, color=color, label=str(level))

        plt.legend(loc='best', bbox_to_anchor=(1, 1), borderaxespad=0)
    plt.tight_layout() 
    plt.xlabel(x)
    plt.ylabel(f'mean({y})')
    plt.show()


# In[42]:


display(test_set['visitor_location_country_id'].value_counts(normalize=True))
display(train_set['visitor_location_country_id'].value_counts(normalize=True))


# In[44]:


ordinal_scatter(train_set, 'position', 'booking_bool', control='random_bool')


# - Overall, interest increases as the historical price increases, however, there is a threshold at which interest decreases again. This indicates that in general higher prices attract higher interest, as long as it is not too expensive, nothing surprising
# - The hotel star-rating follows the same pattern, hotels with higher stars garner more interest, however 5-star hotels have less interest
# - For actual bookings, we see that the patterns are the same. However, booking frequency decreases much more drastically for extremely highly priced hotels than interest. This also makes sense, when it comes to actually booking the hotel, far fewer people will actually book the very expensive one. 

# When adding the booking window as a third variable, we observe the following differences to the general trends:

# In[53]:


ordinal_scatter(train_set, 'prop_starrating', 'click_bool', 'srch_booking_window', scatter=False, bin_control=True)
ordinal_scatter(train_set, 'prop_starrating', 'booking_bool', 'srch_booking_window', scatter=False, bin_control=True)
ordinal_scatter(train_set, 'prop_log_historical_price', 'click_bool', 'srch_booking_window', scatter=False, bin_x=True, bin_control=True)
ordinal_scatter(train_set, 'prop_log_historical_price', 'booking_bool', 'srch_booking_window', scatter=False, bin_x=True, bin_control=True)


# - For 1-star hotels, a shorter booking window seems to increase interest more than longer windows
# - For 5-star hotels, a shorter booking window decreases interest more than longer windows
#     - People who search more in advance are more interested in higher-grade hotels
# - For booking, the above does not apply as strongly. But we can see that for big booking windows, bookings are less than for smaller booking windows. This makes sense: When the intended stay is far away, you are not booking yet, but as it approaches, you start booking. This is not very informative ofc.
