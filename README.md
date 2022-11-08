
# Project Structure

- `env.yml` Requirements for conda environnement
- `make_dataset` Download and pre-process wikipedia dump. It takes some time for executing
- `loader` Code for loading couples word/embedding from wikipedia dump
- `dataset` folder containing pre-processed dataset and index files. Automatically generated
- `model` pytorch model
- `w2v` implementation of word2vec model