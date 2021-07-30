from utils.utils import load_cctld_list, get_labels_counts_map, set_logger
import pyarrow.parquet as pq
import pandas as pd
from os import listdir
import math
from utils.utils import CCTLD
from urllib.parse import urlparse
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
import pickle
from tqdm import tqdm
import hdbscan
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import scipy.sparse
from scipy.sparse import hstack


logger = set_logger("./log/preprocessing.log")


def load_data(path, mode):
    """Load all parquet files in the input path into a pandas DataFrame."""
    # Read all the parquet files
    filenames = [
        path+filename for filename in listdir(path) if filename.endswith('.parquet')]
    # The input path specified by the user must contain at least one parquet file
    if not filenames:
        raise ValueError('No Parquet file in the specified path')
    # Load the parquet files into a DataFrame
    df = pd.concat([pq.read_table(filename).to_pandas()
                   for filename in filenames], ignore_index=True)
    if "train" in mode and list(df.columns) != ['url', 'target', 'day']:
        raise ValueError(
            'Please make sure your train data is in the required format (columns must be named : url target day)')
        logger.error('Failed to load dataset. See error message')
    if mode == "test" and list(df.columns) != ['url', 'day']:
        raise ValueError(
            'Please make sure your test data is in the required format (columns must be named : url day)')
        logger.error('Failed to load dataset. See error message')

    logger.info('Data loaded successfully : {} samples and {} columns'.format(
        df.shape[0], df.shape[1]))

    return df


def vectorize_url(df, mode):
    """ Vectorize url using TF-IDF."""
    new_df = df.copy()
    new_df['clean_url'] = df['url'].apply(lambda x: x.split(
        '://')[1])  # we remove (http(s)://) as all urls start with it and it just adds noise to the n-grams

    if mode == "train":
        vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(
            1, 3), norm='l2', max_df=0.8, max_features=10000)
        vectorizer.fit(new_df['clean_url'])
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
    else:
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)

    url_array = vectorizer.transform(new_df['clean_url'])

    return url_array


def categorize_tld(tld):
    """ Return the category of the top level domain."""
    # Put the most common tlds into separate categories
    if tld in ['com', 'fr', 'net', 'org']:
        return tld
    # Most of the other categories are country code categories (other than french)
    if tld in CCTLD:  # Not french country code tld
        return 'cctld'
    
    return 'other'


def categorize_subdomain(subdomain):
    """ Return the category of the subdomain."""
    if subdomain in ['www', 'm', 'forum', 'context', 'annuaire', 'clients', 'no_subdomain', 'jeu', 'mots-croises']:
        return subdomain
    if subdomain == 'fr':
        return 'fr_subdomain'
    if subdomain in ['dictionnaire', 'mobile-dictionary', 'dict']:
        return 'dict'
    
    return 'other_subdomain'


def get_labels_vector(df, nb_labels=100):
    hostnames = list(df['hostname'].value_counts()[:nb_labels].index.tolist())
    hostname2vec = {hostname: [] for hostname in hostnames}
    # first fill each hostname with all the labels it takes (with repetition)
    for idx, row in df.iterrows():
        if row['hostname'] in hostnames:
            hostname2vec[row['hostname']
                         ] = hostname2vec[row['hostname']]+list(row['target'])
    # transform each value (initially containing a list of labels) to a vector defined on each dimension
    # by the occurence of the label with that hostname/ occurence of the hostname in the dataset
    # In other words, the coordinate of each dimension is the ratio of how often a label is used with the hostname
    labels_counts_map = get_labels_counts_map(df)
    labels_list = list(labels_counts_map.keys())

    for k, v in tqdm(hostname2vec.items()):
        hostname2vec[k] = [
            v.count(str(i))/len(df[df['hostname'] == k]) for i in labels_list]

    return hostname2vec


def select_rare_labels(df, max_freq=20):
    selected_labels = []
    labels_counts_map = get_labels_counts_map(df)
    for k, v in labels_counts_map.items():
        if v <= max_freq:
            selected_labels.append(k)
    return selected_labels


def extract_features(df, mode, update_map):
    """Return a DataFrame with the new engineered features."""
    # Day is a circular variable, hence we represent it using 2 new features :
    new_df = df.copy()
    new_df['x_day'] = df['day'].apply(lambda x: math.cos(2*math.pi*x/30))
    new_df['y_day'] = df['day'].apply(lambda x: math.sin(2*math.pi*x/30))
    ##### Extract new features from the URL ####

    # Extract basic features
    new_df['is_secure'] = df['url'].apply(
        lambda x: x.startswith('https')).astype(int)
    new_df['has_htm_ext'] = df['url'].apply(
        lambda x: bool(re.match('.*\.html?$', x))).astype(int)
    new_df['has_get_tag'] = df['url'].apply(lambda x: '=' in x).astype(int)
    median_length = df['url'].apply(len).median()
    new_df['has_long_url'] = df['url'].apply(
        lambda x: len(x) > median_length).astype(int)
    new_df['has_digit'] = df['url'].apply(
        lambda x: any(i.isdigit() for i in x)).astype(int)

    # Extract and categorize the tld (top-level-domain)
    new_df['tld'] = df['url'].apply(lambda x: urlparse(x).hostname.split(
        '.')[-1])  # We consider only the last part of the tld
    new_df['tld_category'] = new_df['tld'].apply(categorize_tld)

    # Extract and categorize the subdomain
    # We suppose that a hostname containing at least 3 parts has a subdomain(we care only about the first subdomain).
    # This is not always correct
    # The only consequence of this is that the (few) uncorrectly identified subdomains will fall into the category "other" (later)
    # instead of the category "no_subdomain" which shouldn't be a big deal
    new_df['subdomain'] = new_df['url'].apply(lambda x: urlparse(
        x).hostname.split('.')[0] if x.count('.') >= 2 else 'no_subdomain')
    new_df['subdomain_category'] = new_df['subdomain'].apply(
        categorize_subdomain)

    # Extract and categorize the hostname
    # We will cluster the 100 most used hostnames according to the labels in the urls they were in.
    new_df['hostname'] = new_df['url'].apply(lambda x: urlparse(x).hostname)
    if mode == "train" and update_map != -1:
        nb_hostnames = 2  # We categorize the most frequent 400 hostnames
        hostname2vec = get_labels_vector(new_df, nb_hostnames)
        df_vecs = pd.DataFrame(hostname2vec.values())
        clusterer = hdbscan.HDBSCAN(metric='manhattan')
        clusterer.fit(df_vecs)
        hostname_category_map = dict(zip(list(new_df['hostname'].value_counts(
        )[:nb_hostnames].index.tolist()), list(map(int, list(clusterer.labels_)))))
        json.dump(hostname_category_map, open(
            'hostname_category_map.json', 'w'))

    else:
        hostname_category_map = json.load(
            open('hostname_category_map.json', 'r'))

    new_df['hostname_category'] = new_df['hostname'].apply(
        lambda x: hostname_category_map[x] if x in hostname_category_map else -2).astype(str)

    # OneHotEncode all the categorical columns
    categorical_cols = ['tld_category',
                        'subdomain_category', 'hostname_category']
    if mode == "train":
        # The way we categorize the columns shouldn't result in any unknow category
        encoder = OneHotEncoder(handle_unknown="error", sparse=False)
        encoder.fit(new_df[categorical_cols])
        with open("encoder.pkl", "wb") as f:
            pickle.dump(encoder, f)
    else:
        with open('encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
    ohe_array = encoder.transform(new_df[categorical_cols])
    ohe_df = pd.DataFrame(ohe_array)

    columns_to_drop = ['url', 'day', 'tld',
                       'subdomain', 'hostname'] + categorical_cols
    new_df = new_df.drop(columns_to_drop, axis=1)
    new_df.reset_index(drop=True, inplace=True)
    ohe_df.reset_index(drop=True, inplace=True)
    new_df = pd.concat([new_df, ohe_df], axis=1)

    return new_df


def transform_df(df, mode, update_map):
    """ Transform the original DataFrame to the classifier's input format."""
    url_array = vectorize_url(df, mode)

    transformed_df = extract_features(df, mode, update_map)
    y = None
    if mode == "train":
        rare_labels = select_rare_labels(df)
        df['target'] = df['target'].apply(
            lambda x: [label for label in x if label not in rare_labels])
        y = df['target']
        multi_label_binarizer = MultiLabelBinarizer().fit(y)
        with open('multi_label_binarizer.pkl', 'wb') as f:
            pickle.dump(multi_label_binarizer, f)
        y = multi_label_binarizer.transform(y)
        transformed_df = transformed_df.drop('target', axis=1)

    transformed_df = scipy.sparse.csr_matrix(
        transformed_df.astype(float).values)
    X = hstack((url_array, transformed_df))
    return X, y
