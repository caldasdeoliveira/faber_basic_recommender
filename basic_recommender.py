import pandas as pd
import numpy as np

FINAL_PRODUCT=True
NUM_OF_RESULTS=10

def train_recommender(train_df, target_id):
    movie_avg_score={}
    movie_list= list(set(train_df.product_productid.unique())-set(train_df.loc[train_df['review_userid'] == target_id].product_productid))
    for prod_id in movie_list:
        movie_table=train_df.loc[train_df['product_productid'] == prod_id].review_score
        score=movie_table.mean()
        movie_avg_score.update({prod_id: score})
        #print(test)

    movie_list.sort(key=movie_avg_score.__getitem__)
    if len(movie_list)>NUM_OF_RESULTS:
        return movie_list[0:NUM_OF_RESULTS]
    else:
        return movie_list


if FINAL_PRODUCT:
    train_file = 'data/movie_reviews.csv'
    train_df = pd.read_csv(train_file)

    target_user_id = input("input user Id for suggested movies: ")

    print(train_recommender(train_df.head(n=1000),target_user_id))

else:
    number_of_data_sets = 1
    data_sets_dir = 'data/sets/'

    for i in range(number_of_data_sets):
        test_file = data_sets_dir + str(i) + '/test.csv'
        train_file = data_sets_dir + str(i) + '/train.csv'
        test_df = pd.read_csv(test_file)
        train_df = pd.read_csv(train_file)
        #print(train_df)

    print(train_recommender(train_df,target_user_id))
