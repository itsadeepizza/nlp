# Sentiment analysis

## Train the classifier

Run `main.py`

## Benchmark

Encoder: mean accuracy 0.64

Aside the encoder, we tested some simpler models as benchmark
- Random Forest Classifier: mean accuracy 0.68
- Linear Regression with L1 loss : mean accuracy 0.65
- Most Frequent class : mean accuracy 0.45


[UmBERTo](https://aclanthology.org/2021.wassa-1.8.pdf): mean accuracy 0.82


## Dataset :
We used the following dataset
**feelit (ITA)** :

[FEEL-IT: Emotion and Sentiment Classification for the Italian Language](https://aclanthology.org/2021.wassa-1.8) (Bianchi et al., WASSA 2021)

Following dataset have been considered, but not used:
**sentipolc (ITA)**:
http://www.di.unito.it/~tutreeb/sentipolc-evalita16/data.html

**sentiment140 (EN)**
http://help.sentiment140.com/for-students/