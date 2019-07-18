import pandas as pd
import numpy as np
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description='DL Sentiment Training')
    parser.add_argument('--traintest_file', dest='traintest_file', default='data/traintest_course_comments.csv', type=str, action='store', help='Train and test course comments CSV file')
    parser.add_argument('--embs_file', dest='embs_file', default='data/embs_course_comments.csv', type=str, action='store', help='Embeddings course comments CSV file')
    parser.add_argument('--entire_file', dest='entire_file', default='data/entire_course_comments.csv', type=str, action='store', help='Original course comments CSV file')
    parser.add_argument('--score_field', dest='score_field', default='learner_rating', type=str, action='store', help='Field title for scores in CSV file')
    parser.add_argument('--samples_per_class', dest='samples_per_class', default=6500, type=int, action='store', help='Samples per class')

    args = parser.parse_args()

    print('Loading entire comments set')
    reviews = pd.read_csv(args.entire_file)
    print('Found', len(reviews.index), 'comments')

    samples_per_class = args.samples_per_class
    balanced_reviews = []

    for label in np.unique(reviews[args.score_field].values):
        if len(reviews[reviews[args.score_field] == label].index) > args.samples_per_class:
            print('Sample for label', label, 'that has', len(reviews[reviews[args.score_field] == label].index), 'instances')
            balanced_reviews.append(reviews[reviews[args.score_field] == label].sample(n=samples_per_class, random_state=0))
        else:
            print('Discarded label', label, 'that has', len(reviews[reviews[args.score_field] == label].index), 'instances')
    reviews_balanced = pd.concat(balanced_reviews)

    reviews_balanced.to_csv(args.traintest_file)
    print('Sampled', len(reviews_balanced.index), 'train/test comments')

    review_others = pd.merge(reviews, reviews_balanced, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    review_others.to_csv(args.embs_file)
    print('Sampled', len(review_others.index), 'embs comments')

if __name__ == "__main__":
    main()

