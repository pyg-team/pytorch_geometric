import gzip
import pickle
import numpy as np
import scipy.sparse as sp
import random as rd
import math
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.linear_model import LogisticRegression
import sklearn.neural_network as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, accuracy_score, f1_score, recall_score
import xgboost as xgb

from run_ours import pos_neg_split, undersample
rd.seed(1)


def parse(path):
	g = gzip.open(path, 'rb')
	for l in g:
		yield eval(l)


def getdict(path):

	reviews = {}
	for d in parse(path):
		review = list(d.values())
		if review[0] not in reviews.keys():
			reviews[review[0]] = []
		reviews[review[0]].append(d)
	return reviews


def date_diff(t1, t2):

	date1 = datetime.fromtimestamp(t1)
	date2 = datetime.fromtimestamp(t2)

	return abs((date1-date2).days)


def star_judge(reviews1, reviews2):

	flag = False
	for r1 in reviews1:
		for r2 in reviews2:
			if r1[0] == r2[0]:
				flag = True
				break
	return flag


def time_judge(reviews1, reviews2, diff=7):

	flag = False
	for r1 in reviews1:
		for r2 in reviews2:
			if date_diff(r1[1], r2[1]) <= diff:
				flag = True
				break
	return flag


def time_entropy(timestamps):

	dates = [datetime.fromtimestamp(stamp) for stamp in timestamps]

	unique_years = list(set([date.year for date in dates]))

	no_date = {year: 0 for year in unique_years}

	for date in dates:
		no_date[date.year] += 1

	date_ratio = [d/len(dates) for d in no_date.values()]

	entropy = -sum([ratio*math.log(ratio) for ratio in date_ratio])

	return entropy


def cal_rating(new_reviews):

	no_rating = sp.lil_matrix((len(new_reviews), 17))

	for i, u in enumerate(new_reviews):
		ratings = [int(review['overall']) for review in new_reviews[u]]
		# feature [2-11]
		vector = [ratings.count(r + 1) for r in range(5)] + [ratings.count(r + 1) / len(ratings) for r in range(5)]
		# feature [12-13]
		vector += [vector[5] + vector[6]] + [vector[8] + vector[9]]
		# feature [14]
		entropy = -sum([ratio*math.log(ratio+1e-5) for ratio in vector[5:10]])
		vector += [entropy]
		# feature [15-18]
		vector += [np.median(ratings)] + [max(ratings)] + [min(ratings)] + [np.mean(ratings)]

		no_rating[i, :] = np.reshape(vector, (1, 17))

	return no_rating


def cal_votes(new_reviews):

	no_votes = sp.lil_matrix((len(new_reviews), 12))

	for i, u in enumerate(new_reviews):
		votes = [review['helpful'] for review in new_reviews[u]]
		help = [vote[0] for vote in votes]
		total = [vote[1] for vote in votes]
		unhelp = [vote[1]-vote[0] for vote in votes]
		# feature [19-20]
		vector = [sum(help)] + [sum(unhelp)]
		# feature [21-24]
		vector += [sum(help)/(sum(total)+1e-5)] + [sum(help)/len(votes)] + [sum(unhelp)/(sum(total)+1e-5)] + [sum(unhelp)/len(votes)]
		# feature [25-30]
		vector += [np.median(help)] + [max(help)] + [min(help)] + [np.median(unhelp)] + [max(unhelp)] + [min(unhelp)]

		no_votes[i, :] = np.reshape(vector, (1, 12))

	return no_votes


def cla_dates(new_reviews):

	no_dates = sp.lil_matrix((len(new_reviews), 3))

	for i, u in enumerate(new_reviews):
		timestamps = [review['unixReviewTime'] for review in new_reviews[u]]
		duration = date_diff(max(timestamps), min(timestamps))
		# feature [31]
		vector = [duration]
		# feature [32]
		vector += [time_entropy(timestamps)]
		# feature [33]
		vector += [1] if duration == 0 else [0]

		no_dates[i, :] = np.reshape(vector, (1, 3))

	return no_dates


def nltk_sentiment(sentence):

	nltk_sentiment = SentimentIntensityAnalyzer()
	score = nltk_sentiment.polarity_scores(sentence)
	return score


def sentiment(new_reviews):

	user_texts = [' '.join([review['reviewText'] for review in reviews]) for reviews in new_reviews.values()]

	nltk_results = []
	for i, text in enumerate(user_texts):
		print(i)
		result = nltk_sentiment(text)
		if result['compound'] > 0:
			nltk_results.append(1)
		elif result['compound'] < 0:
			nltk_results.append(-1)
		else:
			nltk_results.append(0)

	return nltk_results


def build_graph(new_reviews):

	user_adj = sp.lil_matrix((len(new_reviews), len(new_reviews)))

	# user-product-user
	products = {u: [review['asin'] for review in reviews] for u, reviews in new_reviews.items()}
	for i1, u1 in enumerate(new_reviews):
		print(i1)
		for i2, u2 in enumerate(new_reviews):
			if u1 != u2 and len(list(set(products[u1]) & set(products[u2]))) >= 1:
				user_adj[i1, i2] = 1
				user_adj[i2, i1] = 1

	# # user-star&time-user
	# rating_time = {user: [(review['overall'], review['unixReviewTime']) for review in reviews] for user, reviews in new_reviews.items()}
	#
	# for i1, u1 in enumerate(new_reviews):
	# 	print(i1)
	# 	for i2, u2 in enumerate(new_reviews):
	# 		if u1 != u2 and star_judge(rating_time[u1], rating_time[u2]) and time_judge(rating_time[u1], rating_time[u2]):
	# 			user_adj[i1, i2] = 1
	# 			user_adj[i2, i1] = 1

	# # user-textsim_user
	# user_text = {user: ' '.join([review['reviewText'] for review in reviews]) for user, reviews in new_reviews.items()}
	#
	# all_text = list(user_text.values())
	# vect = TfidfVectorizer(min_df=1, stop_words="english")
	# tfidf = vect.fit_transform(all_text)
	# simi = tfidf * tfidf.T
	# simi_arr = simi.toarray()
	# np.fill_diagonal(simi_arr, 0)
	# flat_arr = simi_arr.flatten()
	# flat_arr.sort()
	# threshold = flat_arr[int(flat_arr.size*0.95)]
	# print(threshold)
	#
	# for i1, u1 in enumerate(new_reviews):
	# 	print(i1)
	# 	for i2, u2 in enumerate(new_reviews):
	# 		if u1 != u2 and simi_arr[i1, i2] >= threshold:
	# 			user_adj[i1, i2] = 1
	# 			user_adj[i2, i1] = 1

	# user-product-star-user
	# products = {u: [review['asin'] for review in reviews] for u, reviews in new_reviews.items()}
	# rating_time = {user: [(review['overall'], review['unixReviewTime']) for review in reviews] for user, reviews in
	# 			   new_reviews.items()}
	# for i1, u1 in enumerate(new_reviews):
	# 	print(i1)
	# 	for i2, u2 in enumerate(new_reviews):
	# 		if u1 != u2 and len(list(set(products[u1]) & set(products[u2]))) >= 1 and star_judge(rating_time[u1], rating_time[u2]):
	# 			user_adj[i1, i2] = 1
	# 			user_adj[i2, i1] = 1

	return user_adj


def build_features(new_reviews):

	features = sp.lil_matrix((len(new_reviews), 36))

	user_ids = list(new_reviews.keys())

	# 1) [0] Number of rated products
	no_prod = [len(reviews) for reviews in new_reviews.values()]
	features[:, 0] = np.reshape(no_prod, (len(no_prod), 1))

	# 2) [1] Length of username
	len_name = [len(reviews[0]['reviewerName']) if 'reviewerName' in reviews[0].keys() else 0 for reviews in new_reviews.values()]
	features[:, 1] = np.reshape(len_name, (len(len_name), 1))

	# 3) [2-11] Number and ratio of each rating level given by a user
	# 4) [12-13] Ratio of positive and negative ratings (4,5)-pos, (1,2)-neg
	# 5) [14] Entropy of ratings -\sum_{r}(percentage_r * \log percentage_{r})
	# 6) [15-18] Median, min, max, and average of ratings
	no_rating = cal_rating(new_reviews)
	features[:, range(2, 19)] = no_rating

	# 7) [19-20] Total number of helpful and unhelpful votes a user gets
	# 8) [21-24] The ratio and mean of helpful and unhelpful votes
	# 9) [25-30] Median, min, and max number of helpful and unhelpful votes
	no_votes = cal_votes(new_reviews)
	features[:, range(19, 31)] = no_votes

	# 10) [31] Day gap
	# 11) [32] Time entropy
	# 12) [33] Same date indicator
	no_dates = cla_dates(new_reviews)
	features[:, range(31, 34)] = no_dates

	# 13) [34] Feedback summary length
	summ_length = [sum([len(review['summary']) if 'summary' in reviews[0].keys() else 0 for review in reviews]) / len(reviews) for reviews in new_reviews.values()]
	features[:, 34] = np.reshape(summ_length, (len(summ_length), 1))

	# 14) [35] Review text sentiment
	senti_scores = sentiment(new_reviews)
	features[:, 35] = np.reshape(senti_scores, (len(senti_scores), 1))

	return features

if __name__ == '__main__':

	# reviews = getdict('reviews_Musical_Instruments.json.gz')

	# pickle.dump(reviews, open('musical_reviews.pickle', 'wb'))

	with open('musical_reviews.pickle', 'rb') as file:
		all_reviews = pickle.load(file)


	# create ground truth
	user_labels = []
	labeled_reviews = {}
	for u, total in all_reviews.items():
		# print([single[2][0] for single in total])
		# print([single[2][1] for single in total])
		helpful, votes = sum([single['helpful'][0] for single in total]), sum([single['helpful'][1] for single in total])
		if votes >= 20:
			if helpful/votes > 0.8:
				labeled_reviews[u] = total
				user_labels.append(0)
			elif helpful/votes < 0.2:
				labeled_reviews[u] = total
				user_labels.append(1)

	all_user = set(list(all_reviews.keys()))
	label_user = set(list(labeled_reviews.keys()))
	unlabel_user = list(all_user - label_user)

	sampled_users = rd.sample(unlabel_user, int(len(unlabel_user)*0.01))

	sampled_reviews = {user: all_reviews[user] for user in sampled_users}

	new_reviews = {**sampled_reviews, **labeled_reviews}

	# build relation graph
	user_adj = build_graph(new_reviews)
	# sp.save_npz('amz_upsu_adj.npz', user_adj.tocsr())
	exit()
	# construct feature vectors
	# features = build_features(new_reviews)
	# sp.save_npz('amz_features.npz', features.tocsr())

	features = sp.load_npz('Amazon_Dataset/amz_features_36.npz')
	features = features.toarray()

	# filter polluted features
	new_features = features[:, :19]
	new_features = np.hstack([new_features, features[:, 30:]])
	# sp.save_npz('amz_features_25.npz', sp.csr_matrix(new_features))
	# exit()
	labeled_features = new_features[len(sampled_users):]
	user_labels = np.array(user_labels)
	# with open('user_labels.pickle', 'wb') as f:
	# 	pickle.dump(user_labels, f)
	# f.close()
	# exit()