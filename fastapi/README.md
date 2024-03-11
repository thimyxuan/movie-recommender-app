
# Déploiement FastAPI

## Déploiement en local

Créer l'image Docker : 

`$ docker build -t movie-matcher-fastapi`

Créer le container Docker : 

`$ docker run -it -v "$(pwd):/home/app" -p 4000:4000 movie-matcher-fastapi`


## Déploiement avec Heroku

Il est également possible de déployer FastAPI directement via Heroku CLI avec les commandes suivantes. 

Assurez-vous d'être connecté à vos comptes Docker et Heroku :  

`$ heroku login`

`$ docker login --username=<your username> --password=$(heroku auth:token) registry.heroku.com`

`$ heroku container:push web -a YOUR_APP_NAME`

`$ heroku container:release web -a YOUR_APP_NAME`

`$ heroku open -a YOUR_APP_NAME`