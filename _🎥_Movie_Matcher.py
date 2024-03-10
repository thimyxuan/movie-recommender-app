### LIBRAIRIES ###
import streamlit as st
import pandas as pd
import requests
import random
from surprise import Dataset, Reader, SVD
import ast


### CONFIGURATION ###
st.set_page_config(
    page_title="Movie Matcher",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="collapsed"
)


### IMPORT FICHIERS ###
@st.cache_data
def load_tmdb_content():
    tmdb_content = pd.read_csv("fastapi/src/TMDB_content.csv")
    return tmdb_content
tmdb_content = load_tmdb_content()
filtered_tmdb_content = load_tmdb_content()

@st.cache_data
def load_tmdb_providers():
    tmdb_providers = pd.read_csv("fastapi/src/TMDB_providers.csv")
    return tmdb_providers
tmdb_providers = load_tmdb_providers()

@st.cache_data
def load_ratings_updated():
    ratings_updated = pd.read_csv("fastapi/src/Movielens_ratings_updated.csv")
    return ratings_updated
ratings_updated = load_ratings_updated()

@st.cache_data
def load_content_based():
    content_based = pd.read_csv("fastapi/src/TMDB_content_based.csv")
    return content_based
content_based = load_content_based()


### SESSION STATE ###
if 'filtered_tmdb_content' not in st.session_state:
    st.session_state.filtered_tmdb_content = tmdb_content.copy()

if 'recommended_movies' not in st.session_state:
    st.session_state.recommended_movies = None
    st.session_state.recommandation = False
    st.session_state.filtered_recommended_movies = None
        
if 'selected_filters_streaming' not in st.session_state:
    st.session_state.selected_filters_streaming = {}
    
if 'selected_filters' not in st.session_state:
    st.session_state.selected_filters = {}


### FONCTIONS ###
def movie_recommandation(favorite_movies):
    global ratings_updated
    favorite_movies = favorite_movies['favorite_movies']
    
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


def apply_filters(df, filters):
    filtered_df = df.copy()

    if filters:
        for filter_name, filter_values in filters.items():
            if filter_values:
                try:
                    filtered_df[filter_name] = filtered_df[filter_name].fillna('')
                    if filter_name == 'watch_providers':
                        filter_conditions = [filtered_df[filter_name].str.contains(fr'\b{value}\b', case=True) for value in filter_values]
                    else:
                        filter_conditions = [filtered_df[filter_name].str.contains(value, case=False) for value in filter_values]
                    combined_condition = pd.concat(filter_conditions, axis=1).any(axis=1)
                    filtered_df = filtered_df[combined_condition]
                except AttributeError:
                    return filtered_df.head(0)
                    
        filtered_df.reset_index(drop=True, inplace=True)
    
    return filtered_df

def calculate_column_ratios(nb_rows):
    if nb_rows < 5:
        left_padding = (5 - nb_rows) / 2
        right_padding = (5 - nb_rows) / 2
        if (left_padding + right_padding) < (5 - nb_rows):
            left_padding += 0.5
        elif (left_padding + right_padding) > (5 - nb_rows):
            right_padding += 0.5
        column_ratios = [left_padding] + [1] * nb_rows + [right_padding]
        return column_ratios
    return [1] * 5


### HEADER ###
image_paths = {
    'light': "img/light.jpg",
    'dark': "img/dark.jpg"
}
random_theme_mode = random.choice(['light', 'dark'])

left_col, center_col, right_col = st.columns([1, 1, 1])
with center_col: 
    st.image(image_paths[random_theme_mode], use_column_width="auto")


### APPLICATION ###

## INTRODUCTION ###
st.markdown("""
    <div style='text-align:center;'>
        <p style="font-size: 1.2rem;">Bienvenue sur Movie Matcher, votre destination privilégiée pour des recommandations cinématographiques.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")


## SELECTION DES FILMS ###
st.markdown("""
    <div style='text-align:center;'>
        <h2>🌟 Votre sélection cinéphile 🌟</h2>
        <p style="font-size: 1.2rem;">Choisissez jusqu'à 5 films que vous avez appréciés, et nous vous suggérerons une sélection de films qui pourraient vous plaire.</p>
    </div>
""", unsafe_allow_html=True)

tmdb_selection = tmdb_content.loc[:,['tmdb_id', 'title']]
tmdb_selection.loc[-1] = [None, 'Selectionnez un film']
tmdb_selection.index = tmdb_selection.index + 1
tmdb_selection = tmdb_selection.sort_index()
tmdb_selection = tmdb_selection.sort_values(by='title', key=lambda x: x.replace('Selectionnez un film', ''))

columns = st.columns(5)
selected_movies = []
for i, column in enumerate(columns):
    column.markdown(f"""
        <div style='text-align:center;'>
            <p style="font-size: 1.2rem;">Film {i+1}</p>
        </div>
    """, unsafe_allow_html=True)
    movie = column.selectbox(f"Film {i+1}", tmdb_selection['title'], label_visibility='collapsed')
    movie_id = tmdb_selection.loc[tmdb_selection['title'] == movie, 'tmdb_id'].values[0]
    selected_movies.append({"id": movie_id, "title": movie})

st.markdown("")

## Bouton recommandations ##
columns = st.columns([2, 1, 2])
with columns[1]:
    button_recommandations = st.button("🤖 Générer les recommandations 🤖", help = "Cliquez ici pour générer les recommandations", type = 'primary')

if button_recommandations:
    ## FastAPI + Filtres ##
    selected_movies_list = [item["id"] for item in selected_movies if item.get("id") is not None]
    if not selected_movies_list:
        st.markdown("""
            <div style='text-align:center;'>
                <p style="font-size: 1.2rem;">La liste est vide, veuillez renseigner un où plusieurs films</p>
            </div>
        """, unsafe_allow_html=True)
    else:      
        data = {'favorite_movies': selected_movies_list}
        success = False
        try:
            api_urls = [
                'http://movie-matcher-fastapi-1:4000/predict',
                'https://movie-matcher-fastapi-6b7d32444024.herokuapp.com/predict',
                # 'https://moviematcher-fastapi.onrender.com/predict'
            ]
            headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
            for api_url in api_urls:
                try:
                    response = requests.post(api_url, headers=headers, json=data)
                    if response.status_code == 200:
                        result_movies = pd.DataFrame(response.json())
                        success = True 
                        break
                except requests.RequestException as e:
                    pass
        except Exception as e:
            pass
        if not success:
            result_movies = pd.DataFrame(movie_recommandation(data))

        recommended_movies = pd.merge(result_movies, tmdb_content, on='tmdb_id', how='left')
        st.session_state.recommended_movies = recommended_movies
        st.session_state.filtered_recommended_movies = recommended_movies.copy()
        st.session_state.recommandation = True

if st.session_state.recommandation:
    st.markdown("""
        <div style='text-align:center;'>
            <p style="font-size: 1.2rem;">Les recommandations ont été générées.</p>
        </div>
    """, unsafe_allow_html=True)
    
st.markdown("---")


### FILTRES ###
st.markdown("""
    <div style='text-align:center;'>
        <h2>🔍 Votre sélection filtrée 🔍</h2>
        <p style="font-size: 1.2rem;">Nous vous recommandons de commencer par explorer les recommandations sans appliquer de filtres.</p>
        <p style="font-size: 1.2rem;">Affinez ensuite votre choix en explorant divers filtres pour trouver le film qui correspond parfaitement à vos préférences.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("")

## Initialisation des filtres ##
if st.session_state.recommandation:
    filters_genres = st.session_state.recommended_movies['genres'].str.split(',').explode('genres').str.strip().sort_values().unique() 
    filters_keywords = st.session_state.recommended_movies['keywords'].dropna().str.split(',').explode('keywords').str.strip().sort_values().unique()
    filters_year = st.session_state.recommended_movies['year'].sort_values(ascending = False).unique()
    filters_cast = st.session_state.recommended_movies['cast'].dropna().str.split(',').explode('cast').str.strip().sort_values().unique()
    filters_director = st.session_state.recommended_movies['director'].sort_values().unique()
    filters_streaming = tmdb_providers[tmdb_providers['provider_id'].astype(str).isin(
        st.session_state.recommended_movies['watch_providers'].dropna().str.split(',').explode('watch_providers').str.strip().unique()
        )]['provider_name'].sort_values()
    
else:
    filters_genres = tmdb_content['genres'].str.split(',').explode('genres').str.strip().sort_values().unique() 
    filters_keywords = tmdb_content['keywords'].dropna().str.split(',').explode('keywords').str.strip().sort_values().unique()
    filters_year = tmdb_content['year'].sort_values(ascending = False).unique()
    filters_cast = tmdb_content['cast'].dropna().str.split(',').explode('cast').str.strip().sort_values().unique()
    filters_director = tmdb_content['director'].sort_values().unique()
    filters_streaming = tmdb_providers[tmdb_providers['provider_id'].astype(str).isin(
        tmdb_content['watch_providers'].dropna().str.split(',').explode('watch_providers').str.strip().unique()
        )]['provider_name'].sort_values()

columns = st.columns(3)
with columns[0]:
    st.markdown("""
        <div style='text-align:center;'>
            <p style="font-size: 1.2rem;">Genres</p>
        </div>
    """, unsafe_allow_html=True)
    st.session_state.selected_filters['genres'] = st.multiselect('Sélectionnez le(s) genre(s)', filters_genres, key = 'filter_genres')
    
with columns[1]:
    st.markdown("""
        <div style='text-align:center;'>
            <p style="font-size: 1.2rem;">Mots-clés</p>
        </div>
    """, unsafe_allow_html=True)
    st.session_state.selected_filters['keywords'] = st.multiselect('Sélectionnez le(s) mot(s)-clé(s)', filters_keywords, key = "filter_keywords")
    
with columns[2]:
    st.markdown("""
        <div style='text-align:center;'>
            <p style="font-size: 1.2rem;">Années</p>
        </div>
    """, unsafe_allow_html=True)
    st.session_state.selected_filters['year'] = st.multiselect('Sélectionnez l\'année ou les années', filters_year, key = "filter_year")

st.markdown("")

## Filtres column 2 ##

columns = st.columns(3)
with columns[0]:
    st.markdown("""
        <div style='text-align:center;'>
            <p style="font-size: 1.2rem;">Acteurs</p>
        </div>
    """, unsafe_allow_html=True)
    st.session_state.selected_filters['cast'] = st.multiselect('Sélectionnez le(s) acteur(s)', filters_cast, key = "filter_cast")
    
with columns[1]:
    st.markdown("""
        <div style='text-align:center;'>
            <p style="font-size: 1.2rem;">Réalisateur</p>
        </div>
    """, unsafe_allow_html=True)
    st.session_state.selected_filters['director'] = st.multiselect('Sélectionnez le réalisateur', filters_director, key = "filter_director")
    
with columns[2]:
    st.markdown("""
        <div style='text-align:center;'>
            <p style="font-size: 1.2rem;">Plateformes de streaming</p>
        </div>
    """, unsafe_allow_html=True)
    st.session_state.selected_filters['watch_providers'] = st.multiselect('Sélectionnez le(s) plateforme(s) de streaming', filters_streaming, key = "filter_streaming")
    st.session_state.selected_filters['watch_providers'] = tmdb_providers[tmdb_providers['provider_name'].isin(st.session_state.selected_filters['watch_providers'])]['provider_id'].astype(str).tolist()
    
st.markdown("")
st.markdown("---")


### AFFICHER FILMS ###
columns = st.columns([3, 1, 3])
with columns[1]:
    button_affichage = st.button("🎥 Afficher les films 🎥", help = "Cliquez ici pour afficher les films recommandés", type = 'primary')

if button_affichage:
    ## Application des filtres ##
    if st.session_state.recommandation:
        if st.session_state.selected_filters or all(not st.session_state.selected_filters[key] for key in st.session_state.selected_filters):
            st.session_state.filtered_recommended_movies = apply_filters(st.session_state.recommended_movies, st.session_state.selected_filters)
            nb_rows = len(st.session_state.filtered_recommended_movies)
            
        data_affichage = st.session_state.filtered_recommended_movies

    elif not st.session_state.recommandation:
        if st.session_state.selected_filters or all(not st.session_state.selected_filters[key] for key in st.session_state.selected_filters):
            st.session_state.filtered_tmdb_content = apply_filters(tmdb_content, st.session_state.selected_filters)
            nb_rows = len(st.session_state.filtered_tmdb_content)

        else:
            nb_rows = len(st.session_state.filtered_tmdb_content)
            
        data_affichage = st.session_state.filtered_tmdb_content
            
    ## Affichage films recommandés ##
    if nb_rows == 0:
        st.markdown("""
            <div style='text-align:center;'>
                <h2>🎞️ Nos recommandations 🎞️</h2>
                <p style="font-size: 1.2rem;">Aucun film disponible. Veuillez relancer la recommandation ou retirer des filtres.</p>
            </div>
        """, unsafe_allow_html=True)    
        
    else:
        st.markdown("""
            <div style='text-align:center;'>
                <h2>🎞️ Nos recommandations 🎞️</h2>
                <p style="font-size: 1.2rem;">Voici quelques-uns des films que nous vous recommandons.</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("")
                
        ## Affichage titre films ##
        if nb_rows >= 5:
            columns = st.columns(5)
            columns_range = [i for i in range(0, 5)]
        else:
            columns = st.columns(calculate_column_ratios(nb_rows))
            columns_range = [i for i in range(1, nb_rows + 1)]
            
        for i in columns_range:
            col = columns[i]
            if nb_rows >= 5:
                movie_name = data_affichage['title'][i]
            else:
                movie_name = data_affichage['title'][i-1]
            col.markdown(f"""
                <div style='text-align:center;'>
                    <p style="font-size: 1rem;">{movie_name}</p>
                </div>
            """, unsafe_allow_html=True)
        
        ## Affichage posters films ##
        poster_url_begin = "https://image.tmdb.org/t/p/w500/"
        
        if nb_rows >= 5:
            columns = st.columns(5)
            columns_range = [i for i in range(0, 5)]
        else:
            columns = st.columns(calculate_column_ratios(nb_rows))
            columns_range = [i for i in range(1, nb_rows + 1)]
            
        for i in columns_range:
            col = columns[i]
            if nb_rows >= 5:
                full_poster_url = poster_url_begin + data_affichage['poster_path'][i]
            else:
                full_poster_url = poster_url_begin + data_affichage['poster_path'][i-1]
            col.image(full_poster_url, use_column_width="auto")
            
        ## Affichage streaming films ##
        if nb_rows >= 5:
            columns = st.columns(5)
            columns_range = [i for i in range(0, 5)]
        else:
            columns = st.columns(calculate_column_ratios(nb_rows))
            columns_range = [i for i in range(1, nb_rows + 1)]
            
        for i in columns_range:
            col = columns[i]
            col.markdown(f"""
                <div style='text-align:center;'>
                    <p style="font-size: 1rem;">Plateforme de streaming :</p>
                </div>
            """, unsafe_allow_html=True)
                      
            if nb_rows >= 5:
                cell = f'{i}'
            else:
                cell = f'{i-1}'
                
            cell = int(cell)
            watch_providers = data_affichage['watch_providers'][cell]
            
            try:
                providers_list = tmdb_providers[tmdb_providers['provider_id'].astype(int).isin([int(provider_id.strip('"')) for provider_id in watch_providers.split(',')])]['provider_name'].to_list()
                providers = ''
                for provider in providers_list:
                    providers += "- " + provider + "\n"
                col.markdown(providers)
                
            except AttributeError:
                if pd.isna(watch_providers):
                    col.markdown(f"""
                        <div style='text-align:center;'>
                            <p style="font-size: 1rem;">Aucun streaming en abonnement disponible</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    providers_list = tmdb_providers[tmdb_providers['provider_id'].astype(int) == int(watch_providers)]['provider_name'].to_list()
                    providers = ''
                    for provider in providers_list:
                        providers += "- " + provider + "\n"
                    col.markdown(providers)
                    
            except Exception as e:
                col.markdown(f"""
                    <div style='text-align:center;'>
                        <p style="font-size: 1rem;">Aucun streaming en abonnement disponible</p>
                    </div>
                """, unsafe_allow_html=True)


st.markdown("---")


### FOOTER ###
st.markdown("""
    <div style='text-align:center;'>
        <p>
            Powered by <a href='https://streamlit.io/'>Streamlit</a>, <a href='https://www.justwatch.com/'>JustWatch</a>, <a href='https://www.themoviedb.org/'>TMDB</a> & <a href='https://movielens.org/'>MovieLens</a>
        </p>
        <p>
            Voir le code source sur <a href='https://github.com/Clementbroeders/movie-matcher'>GitHub</a>. © 2024 Movie Matcher.
        </p>
    </div>
""", unsafe_allow_html=True)