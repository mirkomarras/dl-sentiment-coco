# Deep Learning Adaptation with Word Embeddings for Sentiment Analysis on Online Course Reviews
[![Build Status](https://travis-ci.org/pages-themes/cayman.svg?branch=master)](https://travis-ci.org/pages-themes/cayman)
[![GitHub version](https://badge.fury.io/gh/boennemann%2Fbadges.svg)](http://badge.fury.io/gh/boennemann%2Fbadges)
[![Dependency Status](https://david-dm.org/boennemann/badges.svg)](https://david-dm.org/boennemann/badges)
[![Open Source Love](https://badges.frapsoft.com/os/gpl/gpl.svg?v=102)](https://github.com/ellerbrock/open-source-badge/)

Online educational platforms are enabling learners to consume a great variety of content and share opinions on their learning experience. The analysis of the sentiment behind such a collective intelligence represents a key element for supporting both instructors and learning institutions on shaping the offered educational experience. Combining Word Embedding representations and Deep Learning architectures has made possible to design Sentiment Analysis systems able to accurately measure the text polarity on several contexts. However, the application of such representations and architectures on educational data still appears limited. Therefore, considering the over-sensitiveness of the emerging models to the context where the training data is collected, conducting adaptation processes that target the e-learning context becomes crucial to unlock the full potential of a model. In this chapter, we describe a Deep Learning approach that, starting from Word Embedding representations, measures the sentiment polarity of textual reviews posted by learners after attending online courses. Then, we demonstrate how Word Embeddings trained on smaller e-learning-specific resources are more effective with respect to those trained on bigger general-purpose resources. Moreover, we show the benefits achieved by combining Word Embeddings representations with Deep Learning architectures instead of common Machine Learning models. We expect that this chapter will help stakeholders to get a clear view and shape the future research on this field. 

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

Split your comments in train/test comments and embedding generation comments. The entire dataset should be a standard
csv file. The csv file should include a field that represents the score associated to the comment. Please download a
sample dataset together with sample splitted files from [this link](). Below you can find a sample splitting command: 
```
$ python comment_splitter.py 
--entire_file "data/entire_course_comments.csv" 
--score_field "learner_rating" 
--traintest_file "data/traintest_course_comments.csv" 
--embs_file "data/embs_course_comments.csv" 
--samples_per_class 6500
```
Create context-specific embeddings from the embedding generation file. Please download sample context specific embeddings from 
[this link](). Below you can find a sample generation command:
```
...
```
Train and test your model from context-specific embeddings and train/test comments files. Please download sample models 
and results from [this link](). Below you can find a sample train/test command:
```
python score_trainer_tester.py 
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

## Contributing
We welcome contributions. Feel free to file issues and pull requests on the repo and we will address them as we can.

For questions or feedback, contact us at [{danilo_dessi, fenu, mirko.marras, diego.reforgiato}@unica.it](http://).

## Citations
If you use this source code in your research, please cite the following entries.

```
Dessì, D., Dragoni, M., Fenu, G., Marras, M., Reforgiato, D. (2019). 
Deep Learning Adaptation with Word Embeddings for Sentiment Analysis on Online Course Reviews. 
Book on Deep Learning based Approaches for Sentiment Analysis. 
Springer.
```

```
Dessì, D., Dragoni, M., Fenu, G., Marras, M., Reforgiato, D. (2019). 
Evaluating Neural Word Embeddings Created from Online Course Reviews for Sentiment Analysis. 
Proceedings of the 34th ACM/SIGAPP Symposium On Applied Computing (SAC2019). 
Springer.
```

## Credits and License
Copyright (C) 2019 by the [Department of Mathematics and Computer Science](https://www.unica.it/unica/it/dip_matinfo.page) at [University of Cagliari](https://www.unica.it/unica/).

This source code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for details.

You should have received a copy of the GNU General Public License along with this source code. If not, go the following link: http://www.gnu.org/licenses/.

