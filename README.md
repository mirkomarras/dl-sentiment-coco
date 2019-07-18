# Deep Learning Adaptation with Word Embeddings for Sentiment Analysis on Online Course Reviews
[![Build Status](https://travis-ci.org/pages-themes/cayman.svg?branch=master)](https://travis-ci.org/pages-themes/cayman)
[![GitHub version](https://badge.fury.io/gh/boennemann%2Fbadges.svg)](http://badge.fury.io/gh/boennemann%2Fbadges)
[![Dependency Status](https://david-dm.org/boennemann/badges.svg)](https://david-dm.org/boennemann/badges)
[![Open Source Love](https://badges.frapsoft.com/os/gpl/gpl.svg?v=102)](https://github.com/ellerbrock/open-source-badge/)

This repository contains the resources outcome of the work "*Deep Learning Adaptation with Word Embeddings for Sentiment Analysis on Online Course Reviews.*".

The code allows you to create a Deep Learning approach that, starting from Word Embedding representations, measures the sentiment polarity of textual reviews posted by learners after attending online courses.

## Installation 

Install Python (>=3.5):
```
$ sudo apt-get update
$ sudo apt-get install python3.5
```
Clone this repository: 
```
$ git clone https://github.com/mirkomarras/dl-sentiment-coco.git
```
Install the requirements:
```
$ pip install -r dl-sentiment-coco/requirements.txt
```

## Usage

#### Prepare data for embedding generation, sentiment prediction training and testing.

The *entire_file* should be a comma-separated csv file including a column *score_field* that lists the scores associated 
to the comments. The script creates two files: (i) *traintest_file* for sentiment training and test and (ii) *embs_file* 
for embedding generation. For sentiment prediction testing, *samples_per_class* samples per class are selected.

Below you can find a sample splitting command: 
```
$ python ./dl-sentiment-coco/code/comment_splitter.py 
--entire_file "./dl-sentiment-coco/data/entire_course_comments.csv" 
--score_field "learner_rating" 
--traintest_file "./dl-sentiment-coco/data/traintest_course_comments.csv" 
--embs_file "./dl-sentiment-coco/data/embs_course_comments.csv" 
--samples_per_class 6500
```

Create a folder *data* and copy the online course review dataset together with its splitted files
available at [this link](https://drive.google.com/file/d/1aZgJAhanQjKV3Gzscx0bjs6_kbImQIKF/view?usp=sharing). 

#### Create context-specific embeddings from the embedding generation file. 

Below you can find a sample embedding generation command:

```
...
```

Create the nested folders *embeddings/specific* and copy the context-specific embeddings available at [this link](https://drive.google.com/file/d/1guu3WT-FaF-keWW1NHBdNRkpvU6g5KRO/view?usp=sharing). 

#### Train and test your model from context-specific embeddings and train/test comments files. 

The *traintest_file* should be a comma-separated csv file including two columns: (i) *comment_field* that lists the
comments and (ii) *score_field* that lists the scores associated to the comments. The script creates models, subsequently 
instantiated with embeddings dictionaries from *embs_dir*, able to assign one of *n_classes* classes to a comment with 
*max_len* words. Each model is trained for *n_epochs* on batches of size *batch_size*, and tested through stratified *n_fold* cross-validation. 

Below you can find a sample train/test command:

```
python ./dl-sentiment-coco/code/score_trainer_tester.py 
--traintest_file "data/traintest_course_comments.csv" 
--comment_field "learner_comment" 
--score_field "learner_rating" 
--max_len 500 
--n_classes 2 
--embs_dir "embeddings/specific/fasttext" 
--n_epochs 20 
--batch_size 512 
--n_fold 5
```

Create two nested folders *models/class2/* and *results/class2* and copy the models and results 
available at [this link](https://drive.google.com/file/d/1pW3XYpfhOvXbXGtQOgu_zC06IupCqKmB/view?usp=sharing). 

## Contributing
We welcome contributions. Feel free to file issues and pull requests on the repo and we will address them as we can.

For questions or feedback, contact us at [{danilo_dessi, fenu, mirko.marras, diego.reforgiato}@unica.it](http://).

## Citations
If you use this source code in your research, please cite the following entries.

```
Dessì, D., Dragoni, M., Fenu, G., Marras, M., Reforgiato, D. (2019). 
Deep Learning Adaptation with Word Embeddings for Sentiment Analysis on Online Course Reviews. 
In: Deep Learning based Approaches for Sentiment Analysis, Springer.
```

```
Dessì, D., Dragoni, M., Fenu, G., Marras, M., & Recupero, D. R. (2019). 
Evaluating Neural Word Embeddings Created from Online Course Reviews for Sentiment Analysis. 
In: Proceedings of the 34th ACM/SIGAPP Symposium on Applied Computing, 2124-2127, ACM.
```

## Credits and License
Copyright (C) 2019 by the [Department of Mathematics and Computer Science](https://www.unica.it/unica/it/dip_matinfo.page) at [University of Cagliari](https://www.unica.it/unica/).

This source code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for details.

You should have received a copy of the GNU General Public License along with this source code. If not, go the following link: http://www.gnu.org/licenses/.

