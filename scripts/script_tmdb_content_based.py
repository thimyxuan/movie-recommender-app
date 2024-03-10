### LIBRAIRIES ###
import pandas as pd
import os
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


### PATHS ###
path = os.getcwd()
project_path = os.path.abspath(os.path.join(path, '..'))


### PREPROCESSING ###
def filter_keywords(x, keyword_counts):
    ''' Remove keywords that appear only one time '''
    return [i for i in x if i in keyword_counts.index]


def to_lemma(nlp, text):
    ''' Transform keyword into its lemma form '''
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc]
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text


def preprocessing_content():
    print('Preprocessing en cours ...')
    nlp = spacy.load('en_core_web_sm')
    
    tmdb_content = pd.read_csv(project_path + '/fastapi/src/TMDB_content.csv')
    tmdb_content = tmdb_content.loc[:,['tmdb_id', 'title', 'genres', 'keywords', 'director', 'cast']]
    keyword_counts = tmdb_content['keywords'].str.split(',').explode('keywords').value_counts().loc[lambda x: x > 1]

    tmdb_content['keywords'] = tmdb_content['keywords'].apply(lambda x: str(x).split(','))
    tmdb_content['keywords'] = tmdb_content['keywords'].apply(lambda x: filter_keywords(x, keyword_counts))
    tmdb_content['keywords'] = tmdb_content['keywords'].apply(lambda x: [to_lemma(nlp, i) for i in x])
    tmdb_content['keywords'] = tmdb_content['keywords'].apply(lambda x: [str.lower(i.replace(' ','')) for i in x])

    tmdb_content['genres'] = tmdb_content['genres'].apply(lambda x: str(x).split(','))
    tmdb_content['genres'] = tmdb_content['genres'].apply(lambda x: [str.lower(i.replace(' ','')) for i in x])

    tmdb_content['cast'] = tmdb_content['cast'].apply(lambda x: str(x).split(','))
    tmdb_content['cast'] = tmdb_content['cast'].apply(lambda x: [str.lower(i.replace(' ','')) for i in x])

    tmdb_content['director'] = tmdb_content['director'].apply(lambda x: [x,x,x])
    tmdb_content['director'] = tmdb_content['director'].apply(lambda x: [str.lower(i.replace(' ','')) for i in x])

    # We keep 6 actors as the main cast:
    if len(tmdb_content['cast']) > 6:
        tmdb_content['cast'] = tmdb_content['cast'].apply(lambda x: x[0:6])

    tmdb_content['soup'] = tmdb_content['genres'] + tmdb_content['keywords'] + tmdb_content['cast'] + tmdb_content['director']
    tmdb_content['soup'] = tmdb_content['soup'].apply(lambda x: ' '.join(x))
    preprocessed_content = tmdb_content.loc[:,['tmdb_id', 'soup']]
    movie_ids = pd.Series(preprocessed_content.index, index=preprocessed_content['tmdb_id']).drop_duplicates()
    
    print('Preprocessing terminé, les données sont prêtes pour l\'application des recommandation.')
    return preprocessed_content, movie_ids


def application_ml(preprocessed_content):
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(preprocessed_content['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    print('Matrice de similarité cosinus créée.')
    return cosine_sim


def get_recommendations(movie_id, cosine_similarity):
    idx = movie_ids[movie_id]
    similarity_scores = list(enumerate(cosine_similarity[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:51]

    movie_indices = [i[0] for i in similarity_scores]
    movie_scores = [i[1] for i in similarity_scores]
    movie_tmdb_id = preprocessed_content['tmdb_id'].iloc[movie_indices]

    content_based = pd.DataFrame(data = {'tmdb_id': movie_tmdb_id, 'score': movie_scores})
    content_based = content_based.reset_index(drop=True)
    return content_based


def application_recommandations(preprocessed_content):
    all_scores = []
    for movie in preprocessed_content['tmdb_id']:
        try:
            scores = get_recommendations(movie, cosine_sim).to_dict(orient='records')
            all_scores.append({'tmdb_id': movie, 'similarities': scores})
        except Exception as e:
            print(f"Error for movie_id {movie}: {e}")
    tmdb_content_based = pd.DataFrame(all_scores)
    tmdb_content_based.to_csv(project_path + '/fastapi/src/TMDB_content_based.csv', index = False)
    print('Recommandations basées sur le contenu créées. Le fichier TMDB_content_based.csv est disponible dans le dossier fastapi/src.')
    return tmdb_content_based


### LANCEMENT DU SCRIPT ###

preprocessed_content, movie_ids = preprocessing_content()

cosine_sim = application_ml(preprocessed_content)

tmdb_content_based = application_recommandations(preprocessed_content)