### LIBRAIRIES ###
import uvicorn
import pandas as pd 
from fastapi import FastAPI
from pydantic import BaseModel
from surprise import Dataset, Reader, SVD
import ast


### APP ###
app = FastAPI()


### LOAD FILES ###
ratings_updated = pd.read_csv('src/Movielens_ratings_updated.csv')
content_based = pd.read_csv('src/TMDB_content_based.csv')


### CLASS ###
class RecommendationRequest(BaseModel):
    favorite_movies: list


### GET ###
@app.get("/")
async def index():
    message = "Bienvenue sur notre API. Ce '/' est l'endpoint le plus simple et celui par défaut. Si vous voulez en savoir plus, consultez la documentation de l'api à '/docs'"
    return message


### POST ###
@app.post("/predict")
async def predict(recommendation_request: RecommendationRequest):
    global ratings_updated
    favorite_movies = recommendation_request.favorite_movies

    ### COLLABORATIVE FILTERING ###
    # Add new user in ratings dataset:
    new_user_id = ratings_updated['userId'].max() + 1
    for movie in favorite_movies:
        newdata = pd.DataFrame([[new_user_id, movie, 5.0]], columns=['userId', 'tmdb_id', 'rating'])
        ratings_updated = pd.concat([ratings_updated, newdata], ignore_index=True)

    # Train model with new data:
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings_updated[['userId', 'tmdb_id', 'rating']], reader)
    best_hyperparams = {'n_factors': 50, 'reg_all': 0.05, 'n_epochs': 20, 'lr_all': 0.005}
    svd = SVD(**best_hyperparams)
    train_set = data.build_full_trainset()
    svd.fit(train_set)

    # Isolate movies the user never saw:
    all_movies = ratings_updated['tmdb_id'].unique().tolist()
    already_seen = ratings_updated[ratings_updated['userId'] == new_user_id]['tmdb_id'].tolist()
    never_seen = [x for x in all_movies if x not in already_seen]

    # Make predictions for new user:
    predictions = []
    movies = []
    for movie in never_seen:
        pred = svd.predict(new_user_id, movie)
        predictions.append(pred.est)
        movies.append(pred.iid)

    # Results collaborative filtering:
    result_collaborative = pd.DataFrame(list(zip(predictions, movies)), columns=['predicted_rating', 'tmdb_id'])
    result_collaborative.sort_values(by='predicted_rating', ascending=False, inplace=True)   
    
    
    ### CONTENT BASED FILTERING ###
    # Filter content and open list of similarities:
    filtered_content_based = content_based[content_based['tmdb_id'].isin(favorite_movies)]
    filtered_content_based.loc[:, 'similarities'] = filtered_content_based['similarities'].apply(ast.literal_eval)

    result_list = []
    for _, row in filtered_content_based.iterrows():
        for entry in row['similarities']:
            result_list.append({
                'tmdb_id': entry['tmdb_id'],
                'score': entry['score']
            })

    result_content = pd.DataFrame(result_list)
    result_content = result_content[~result_content['tmdb_id'].isin(favorite_movies)]

    # Calculate score_content:
    alpha = 0.1  # Poids pour les valeurs en doublon

    result_content = result_content.groupby('tmdb_id').agg({'score': 'sum', 'tmdb_id': 'count'})
    result_content = result_content.rename(columns={'tmdb_id': 'count_duplicates'})
    result_content = result_content.reset_index()
    result_content['score_content'] = result_content.apply(lambda row: row['score'] / row['count_duplicates'] + alpha * row['count_duplicates'] if row['count_duplicates'] > 1 else row['score'], axis=1)
    
    # Results content_based
    result_content.sort_values(by='score_content', ascending=False, inplace=True)
    
    
    ### REGROUPER LES FICHIERS ###
    result = pd.merge(result_content, result_collaborative, on='tmdb_id', how='left')
    
    weight_collaborative = 1  # Poids pour le modèle de filtrage collaboratif
    weight_content = 5  # Poids pour le modèle content-based
    
    average_predicted_rating = result['predicted_rating'].mean()
    result['final_score'] = (weight_collaborative * result['predicted_rating'].fillna(average_predicted_rating) + weight_content * result['score_content'])
    
    # Results
    result.sort_values(by='final_score', ascending=False, inplace=True)
    result = result.loc[:,['tmdb_id', 'final_score']]
    
    return result.to_dict(orient='records')


### RUN APP ###
if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)