# Analyzing Trump Tweets Using Sentiment Analysis

## Resources

Dataset: https://www.kaggle.com/austinreese/trump-tweets <br />
Guides (helped me throughout the project):

1. https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24
2. https://www.kaggle.com/shailaja4247/sentiment-analysis-of-tweets-wordclouds-textblob
3. https://www.kaggle.com/erikbruin/text-mining-the-clinton-and-trump-election-tweets#header
4. https://www.kaggle.com/vyombhatia/trump-tweet-generator-fastai/data

## Setup for developement:

- Setup a python 3.x venv (usually in `.venv`)
  - You can run `./scripts/create-venv.sh` to generate one
- `pip3 install --upgrade pip`
- Install dev requirements `pip3 install -r requirements.dev.txt`
- Install requirements `pip3 install -r requirements.txt`
- `pre-commit install`

## Update versions

`pip-compile --output-file=requirements.dev.txt requirements.dev.in --upgrade`

## Run `pre-commit` locally.

`pre-commit run --all-files`

## Run Docker

> Note: To run all code for this project with the same environment, please use docker

- Build docker image `docker-compose build`
- Then run the docker container `docker-compose up`
