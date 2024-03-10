### LIBRAIRIES ###
import pandas as pd
import requests
import zipfile
from io import BytesIO, StringIO
import os
import warnings
warnings.filterwarnings("ignore")


### PATHS ###
path = os.getcwd()
project_path = os.path.abspath(os.path.join(path, '..'))


### FONCTIONS ###

def download_movielens_data():
    print('Téléchargement des données MovieLens 25m en cours...')
    url_movielens = 'https://files.grouplens.org/datasets/movielens/ml-25m.zip'
    response = requests.get(url_movielens)
    if response.status_code == 200:
        print('Données MovieLens 25m téléchargées avec succès')
        with zipfile.ZipFile(BytesIO(response.content), 'r') as zip_ref:
            zip_file_list = zip_ref.namelist()
            for csv_file_name in zip_file_list:
                if csv_file_name == 'ml-25m/links.csv':
                    with zip_ref.open(csv_file_name) as csv_file:
                        csv_data = csv_file.read().decode('utf-8')
                        movielens_links = pd.read_csv(StringIO(csv_data))
                        print(f"Données chargées avec succès depuis {csv_file_name} en dataframe movielens_links")
                elif csv_file_name == 'ml-25m/ratings.csv':
                    with zip_ref.open(csv_file_name) as csv_file:
                        csv_data = csv_file.read().decode('utf-8')
                        movielens_ratings = pd.read_csv(StringIO(csv_data))
                        print(f"Données chargées avec succès depuis {csv_file_name} en dataframe movielens_ratings")
    else:
        print(f"Échec du téléchargement du fichier. Statut : {response.status_code}")
    
    return movielens_links, movielens_ratings


def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)


def application_weighted_rating(movielens_links):
    # Chargement des données
    tmdb_content = pd.read_csv(project_path + '/fastapi/src/TMDB_content.csv')
    tmdb_content = pd.merge(tmdb_content, movielens_links, left_on='tmdb_id', right_on='tmdbId', how='left')
    tmdb_content = tmdb_content.loc[:,['tmdb_id', 'movieId', 'vote_count', 'vote_average']]
    tmdb_content.rename(columns={'movieId' : 'movielens_id'}, inplace=True)

    # Application du weighted rating IMDB
    vote_counts = tmdb_content[tmdb_content['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = tmdb_content[tmdb_content['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.75)
    weighted_movies = tmdb_content[(tmdb_content['vote_count'] >= m) & (tmdb_content['vote_count'].notnull()) & (tmdb_content['vote_average'].notnull())]
    weighted_movies['vote_count'] = weighted_movies['vote_count'].astype('int')
    weighted_movies['vote_average'] = weighted_movies['vote_average'].astype('int')
    weighted_movies['score_imdb'] = weighted_movies.apply(lambda x: weighted_rating(x, m, C), axis=1)
    nombre_films = weighted_movies.shape[0]
    weighted_movies = weighted_movies.sort_values('score_imdb', ascending=False).head(nombre_films)
    print(f'Vote average : {C}')
    print(f'Nombre de votes minimum : {m}')
    print(f'Nombre de films choisis : {nombre_films}')
    
    return weighted_movies


def update_tmdb_content(weighted_movies):
    tmdb_content = pd.read_csv(project_path + '/fastapi/src/TMDB_content.csv')
    if 'score_imdb' in tmdb_content.columns:
        tmdb_content = tmdb_content.drop('score_imdb', axis=1)
    weighted_movies = weighted_movies.loc[:, ['tmdb_id', 'score_imdb']]
    tmdb_content = pd.merge(weighted_movies, tmdb_content, on='tmdb_id', how='left')
    tmdb_content = tmdb_content.drop_duplicates(subset='tmdb_id', keep='first')
    tmdb_content = tmdb_content.sort_values('score_imdb', ascending=False)
    tmdb_content.to_csv(project_path + '/fastapi/src/TMDB_content.csv', index=False)
    print('Le fichier TMDB_content.csv a été mis à jour avec succès')
    return tmdb_content


def update_movielens_ratings(movielens_ratings, weighted_movies):
    movielens_ratings = pd.merge(weighted_movies, movielens_ratings, left_on='movielens_id', right_on='movieId', how='left')
    movielens_ratings = movielens_ratings.loc[:, ['tmdb_id', 'userId', 'rating']]
    movielens_ratings = movielens_ratings[movielens_ratings['userId'].notnull()]

    # Filtrage des utilisateurs
    user_counts = movielens_ratings['userId'].value_counts()
    borne_inf = 20 # Arbitraire, selon les infos de Movielens
    borne_sup = user_counts[user_counts >= borne_inf].quantile(0.8)
    filtered_users = user_counts[(user_counts >= borne_inf) & (user_counts < borne_sup)].index
    movielens_ratings = movielens_ratings[movielens_ratings['userId'].isin(filtered_users)]

    # Sample d'utilisateurs en fonction du nombre lignes souhaitées
    mediane = movielens_ratings['userId'].value_counts().quantile(0.5)
    nombre_lignes = 250000
    nombre_users = int(nombre_lignes/mediane)
    users_keep = movielens_ratings['userId'].value_counts().sample(nombre_users, random_state = 0).index
    movielens_ratings = movielens_ratings[movielens_ratings['userId'].isin(users_keep)]
    user_mapping = {old_id: new_id for new_id, old_id in enumerate(movielens_ratings['userId'].unique(), start=1)}
    movielens_ratings['userId'] = movielens_ratings['userId'].map(user_mapping)

    movielens_ratings.to_csv(project_path + '/fastapi/src/Movielens_ratings_updated.csv', index=False)
    print('Le fichier movielens_ratings_updated.csv a été créé avec succès.')

    return movielens_ratings


### LANCEMENT DU SCRIPT ###
    
movielens_links, movielens_ratings = download_movielens_data()

weighted_movies = application_weighted_rating(movielens_links)

tmdb_content = update_tmdb_content(weighted_movies)

movielens_ratings = update_movielens_ratings(movielens_ratings, weighted_movies)