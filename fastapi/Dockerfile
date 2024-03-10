FROM continuumio/miniconda3

WORKDIR /home/app

# Install OS Dependencies
RUN apt-get update
RUN apt-get install nano unzip
RUN apt install curl -y

# Install Heroku CLI
RUN curl -fsSL https://get.deta.dev/cli.sh | sh

# Install Python Dependencies
COPY requirements.txt /home/app
RUN conda install -c conda-forge scikit-surprise
RUN pip install --no-cache-dir -r requirements.txt

# Send files to the container
COPY . /home/app/

# Set the default value for the environment variable
ENV PORT=4000

CMD gunicorn app:app --bind 0.0.0.0:$PORT --worker-class uvicorn.workers.UvicornWorker