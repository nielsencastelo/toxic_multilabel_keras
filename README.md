# toxic_multilabel_keras
 Estou usando o keras para classificacao multilabel de comentarios de textos da base de dados 
 Toxic Comment Classification Challenge.
 
 ### Base de dados (TRAIN DATASET)
* Number of data points 159571
* Number data points of type toxic 15294
* Number data points of type severe_toxic 1595
* Number data points of type obscene 8449
* Number data points of type threat 478
* Number data points of type insult 7877
* Number data points of type identity_hate 1405
* Observations in one or more class 35098
* Unclassified observation 124473

 A url da base de dados (https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview)
 
 ![](https://github.com/nielsencastelo/toxic_multilabel_keras/blob/main/bar.png)
 
 #### Word Embedding
 
 Para criacao do Embedding eu utilizo o glove (https://nlp.stanford.edu/projects/glove/)
 
 
 ### Resultados
![](https://github.com/nielsencastelo/toxic_multilabel_keras/blob/main/model_accuracy.png)
![](https://github.com/nielsencastelo/toxic_multilabel_keras/blob/main/model_loss.png)