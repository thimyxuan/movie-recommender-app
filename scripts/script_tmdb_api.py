### LIBRAIRIES ###
import pandas as pd
import requests
from datetime import datetime, timedelta
import gzip
from io import BytesIO
import os
import warnings
warnings.filterwarnings("ignore")


### PATHS ###
path = os.getcwd()
project_path = os.path.abspath(os.path.join(path, '..'))


### FONCTIONS ###
def download_tmdb_daily(number_of_movies= 1000):
    # Get the URL file
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    formatted_date = yesterday.strftime("%m_%d_%Y")
    url_begin = 'http://files.tmdb.org/p/exports/'
    url_end = f'movie_ids_{formatted_date}.json.gz'
    full_url = url_begin + url_end

    # Send a GET request to the URL
    response = requests.get(full_url)
    if response.status_code == 200:
        compressed_data = BytesIO(response.content)
        with gzip.GzipFile(fileobj=compressed_data, mode='rb') as f:
            json_data = f.read().decode('utf-8')
        tmdb_daily = pd.read_json(json_data, lines = True)
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

    # Filter columns
    tmdb_daily = tmdb_daily.loc[:,['id', 'original_title', 'popularity']]
    tmdb_daily = tmdb_daily.sort_values(by=['popularity'], ascending=False).head(number_of_movies)
    print(f'Le fichier tmdb_daily a été téléchargé et filtré avec {number_of_movies} films.')
    return tmdb_daily


def api_request(tmdb_daily, print_interval=100):    
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJhMTFkY2JjYzE4MTFlNWIxOGI3MDg1MTIyOWRiOGYzZSIsInN1YiI6IjY1OTQzNWVlY2U0ZGRjNmQ5MDdlYWQxNSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.5VYKD-qYGgixOfyjsDIR5We_wmJklWml5waulWzQVTA"
    }
    url_start = "https://api.themoviedb.org/3/movie/"
    url_end = '?language=en-US&append_to_response=keywords,credits,watch/providers'

    movie_details_list = []
    movie_keywords_list = []
    movie_credits_list = []
    movie_director_list = []
    movie_providers_list = []
    csv_providers_list = []

    print('Extraction des données API TMDB en cours...')
    
    for i, movie_id in enumerate(tmdb_daily.iloc[:, 0], start=1):
        url = url_start + str(movie_id) + url_end
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            movie_data = response.json()
            
            # Movie details
            movie_details = {
                "tmdb_id": movie_data.get("id", None),
                "title": movie_data.get("title", None),
                "genres": movie_data.get("genres", []),
                "release_date": movie_data.get("release_date", None),
                "vote_average": movie_data.get("vote_average", None),
                "vote_count": movie_data.get("vote_count", None),
                "poster_path": movie_data.get("poster_path", None),
            }
            movie_details_list.append(movie_details)
            
            # Movie keywords
            movie_keywords = {
                "tmdb_id": movie_data.get("id", None),
                "keywords": movie_data.get("keywords", {}).get("keywords", [])
            }
            movie_keywords_list.append(movie_keywords)
            
            # Movie credits
            movie_credits = {
                "tmdb_id": movie_data.get("id", None),
                "cast": []
            }
            cast_data = movie_data.get('credits', {}).get('cast', [])

            for actor in cast_data:
                actor_info = {
                    "name": actor.get("name", None)
                }
                movie_credits["cast"].append(actor_info)
            movie_credits_list.append(movie_credits)

            # Movie director
            crew_data = movie_data.get('credits', {}).get('crew', [])
            for crew_member in crew_data:
                if crew_member.get("job", None) == 'Director':
                    movie_director = {
                        "tmdb_id": movie_data.get("id", None),
                        "director": crew_member.get("name", None)
                    }
                    movie_director_list.append(movie_director)
                    
            # Movie providers
            movie_providers = {
                "tmdb_id": movie_data.get("id", None),
                "watch_providers": []
            }
            
            providers_data = movie_data.get("watch/providers", {}).get("results", {}).get("FR", {}).get("flatrate", [])  
            for provider in providers_data:
                provider_info = {
                    "provider_id": provider.get("provider_id", None)
                }
                movie_providers["watch_providers"].append(provider_info)
            movie_providers_list.append(movie_providers)
            
            
            # Movie providers as CSV
            movie_providers = movie_data.get("watch/providers", {}).get("results", {}).get("FR", {}).get("flatrate", [])    
            for provider_info in movie_providers:
                provider = {
                    "provider_id": provider_info.get("provider_id", None),
                    "provider_name": provider_info.get("provider_name", None),
                    "logo_path": provider_info.get("logo_path", None)
                }
                csv_providers_list.append(provider)
            
            if i % print_interval == 0:
                print(f"Traitement de {i} movie_id sur un total de {len(tmdb_daily.iloc[:, 0])}")    
                
        else:
            print(f"Error fetching details for movie_id: {movie_id}")
            
    print('Toutes les données ont été extraites avec succès.')
    return movie_details_list, movie_keywords_list, movie_credits_list, movie_director_list, movie_providers_list, csv_providers_list


def create_movie_content(movie_details_list, movie_keywords_list, movie_credits_list, movie_director_list, movie_providers_list, csv_providers_list):    
    # Create dataframe df_movie with movie details
    df_movie = pd.DataFrame(movie_details_list)
    df_movie['genres'] = df_movie['genres'].apply(lambda x: [genre['name'] for genre in x]).apply(lambda x: ', '.join(x))
    df_movie['release_date'] = pd.to_datetime(df_movie['release_date'])
    df_movie['year'] = df_movie['release_date'].dt.year
    df_movie.drop(columns=['release_date'], inplace=True)
    df_movie['year'] = df_movie['year'].astype(str)
    def concatener_annees(group): # Fonction pour concatener les titres des films avec les années si doublon
        if len(group) > 1:
            group['title'] = group['title'] + ' (' + group['year'] + ')'
        return group
    df_movie = df_movie.groupby('title').apply(concatener_annees).reset_index(drop=True)

    # Create dataframe df_keywords from movie keywords
    df_keywords = pd.DataFrame(movie_keywords_list)
    df_keywords['keywords'] = df_keywords['keywords'].apply(lambda x: [genre['name'] for genre in x]).apply(lambda x: ', '.join(x))

    # Create dataframe df_credits from movie credits
    df_credits = pd.DataFrame(movie_credits_list)
    df_credits['cast'] = df_credits['cast'].apply(lambda x: [genre['name'] for genre in x]).apply(lambda x: ', '.join(x))
    df_credits['cast'] = df_credits['cast'].apply(lambda x: ', '.join(x.split(', ')[:10]) if isinstance(x, str) else x)

    # Create dataframe df_director from movie director
    df_director = pd.DataFrame(movie_director_list)
    df_director = df_director.drop_duplicates('tmdb_id', keep='first')

    # Create dataframe df_providers from movie providers
    df_providers = pd.DataFrame(movie_providers_list)
    df_providers['watch_providers'] = df_providers['watch_providers'].apply(lambda x: [str(provider['provider_id']) for provider in x]).apply(lambda x: ', '.join(x))
    df_providers['watch_providers'] = df_providers['watch_providers'].replace('', None)


    # Merge all dataframe in one : merged_df
    merged_df = pd.merge(df_movie, df_keywords, on='tmdb_id', how='left')
    merged_df = pd.merge(merged_df, df_credits, on='tmdb_id', how='left')
    merged_df = pd.merge(merged_df, df_director, on='tmdb_id', how='left')
    merged_df = pd.merge(merged_df, df_providers, on='tmdb_id', how='left')


    # Convert to CSV
    merged_df.to_csv(project_path + '/fastapi/src/TMDB_content.csv', index=False)
    print("merged_df a été exporté et enregistré dans le répertoire src/TMDB_content.csv")

    # Create dataframe df_providers_csv from movie providers and export as CSV
    df_providers_csv = pd.DataFrame(csv_providers_list)
    df_providers_csv = df_providers_csv.drop_duplicates('provider_name', keep='first')
    df_providers_csv.to_csv(project_path + '/fastapi/src/TMDB_providers.csv', index=False)
    print("df_providers_csv a été exporté et enregistré dans le répertoire src/TMDB_providers.csv")
    
    return merged_df, df_providers_csv


### LANCEMENT DU SCRIPT ###

tmdb_daily = download_tmdb_daily(number_of_movies=20000)

movie_details_list, movie_keywords_list, movie_credits_list, movie_director_list, movie_providers_list, csv_providers_list = api_request(tmdb_daily, print_interval=100)

merged_df, df_providers_csv = create_movie_content(movie_details_list, movie_keywords_list, movie_credits_list, movie_director_list, movie_providers_list, csv_providers_list)