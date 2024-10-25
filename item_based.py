import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


sample_size = 20000
num_sample = 10

df0 = pd.read_csv('RAW_interactions.csv')
df2 = pd.read_csv('RAW_recipes.csv')


def match_recipes(user_id,recommendations):
    print("Reccommend for", user_id)
    nameid = recommendations.index.tolist()
    for k in nameid:
        print(df2.loc[df2['id'] == k, 'name'].tolist())
    print("end")

def items_based(user_id,df, num_recommendations=5):
    ratings_matrix = df.pivot_table(index='user_id', columns='recipe_id', values='rating', fill_value=0)

    # calculate simility (use cosine)
    item_similarity = cosine_similarity(ratings_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=ratings_matrix.columns, columns=ratings_matrix.columns)


    #check if user id is in dataframe
    if(df[df['user_id'] == user_id].shape[0] > 0):
    
        # rating data set for user
        user_ratings = ratings_matrix.loc[user_id]
    
        #save recommend
        similar_items = pd.Series(dtype=float)

        for item, rating in user_ratings.items():
            if rating > 0:
                #find similary value
                similar_items = pd.concat([similar_items, item_similarity_df[item] * rating])

        # sorting
        similar_items = similar_items.sort_values(ascending=False)
        # Exclude items that have already been rated
        similar_items = similar_items[~similar_items.index.isin(user_ratings[user_ratings > 0].index)]
    
        return similar_items.head(num_recommendations)
    else:
        similar_items = pd.Series(dtype=float)
        return similar_items.head(num_recommendations)


def sampling_items_based(s, num_recommendations=5):
    all_similar_items = pd.Series(dtype=float)
    
    for i in range(s):
        print("sampling . . . ",i+1)
        sample_df = df0.sample(n = sample_size,replace=False)

        #Since not all user IDs are used during the samplimg process, select from the sampled users.
        if(i == 0):
            value = sample_df.iloc[0, 0]
        
        similar_items = items_based(value ,sample_df, num_recommendations)

        
        all_similar_items = pd.concat([all_similar_items , similar_items])
    
    # sorting
    all_similar_items = all_similar_items.sort_values(ascending=False)

    
    all_similar_items = all_similar_items.head(num_recommendations)

    print(all_similar_items)
    match_recipes(value,all_similar_items)




recommendations = sampling_items_based(s = num_sample)




