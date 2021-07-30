# Multi Label Classification

## Setup
1/ Clone the repository
```bash
git clone https://github.com/syri1/Multi-Label-Classification.git && cd Multi-Label-Classification
```

2/ Install the required libraries
```bash
pip install -r requirements.txt
```

**Note** 
- Use Python 3.9.5

- If hdbscan installation throws an error, if you are using conda, you could try to install it with :
```bash
conda install -c conda-forge hdbscan 
```
## Running the code on your own data :
In order to use this project, you need to run the main.py script and specify the appropriate parameters :
* --mode : "train", "test" or "train-test". Default is "train-test"
* --test_size : Relevant only if mode is "train-test". Specifies the test set size. Float between 0 and 1. Default is 0.2.
* --prediction_threshold : Relevant only if mode is "train-test" or "test". Default is 0.27.
* --path : path to the directory containing the parquet file(s). Default is the project's directory.
* --update_hostname_category_map : Relevant only if mode is not "test". keep default if you want to use the provided hostname-category map, or an integer > 0 if you want to reconstruct the map with the provided parameter most frequent hostnames, the higher the parameter, the better the performance is expected to be, and the slower is the map to generate. Default is -1. (Explanation in the comments below)

To reproduce the results stated below, set the variable path to the path of the directory containing the data I was provided. This should re generate all the pkl and predictions files under this directory, or you can start experimenting with your own data

**Note :** 
- if mode is set to "train" or "train-test, the data should be structured under 3 columns : "url", "target", "day". The one hot encoder, the TFIDF vectorizer, the Multi Label Binarizer as well as the model will be saved in the project's directory. 
- The input dataset is expected to have the same format as the provided dataset : no null values, the url follows the regular expression : https?://([a-zA-Z0-9\-]+\.)+[a-zA-Z]+(/|$), the column day is expected to contain integers, and the column target is expected to contain a list of integers.
- if mode is set to "test", the data should be structured under 2 columns : "url", "day". 
- if the mode is set to "test" or "train-test", a predictions parquet file will be generated.




<details>
  <summary>

  ## Comments :
  
  </summary>

- **On the labels :** 
    - The provided dataset uses 1903 distinct labels that take values between 3 and 5904. There's very few labels with twins (share the same occurences) or strong correlation to other labels, so there's no point in trying to reduce the labels space through removing duplicates/strongly correlated labels. 
    - No information was provided about the labels which puts an additional layer of difficulty on the problem. We can't know if the labels are fixed beforehand and if the labeling process consists of manually assigning the subset that applies to each label or the labels are just relevant tags that were present on the website. If it's the latter, then we may have some labels that refer to the same thing or to very close things that appear interchangeably, and this may be a problem. This also makes it unclear if the labels are always restricted to the 1903 ones present in the provided dataset, or if they can take any integer value. To be safe, we suppose it's the latter. It's also worth mentioning that knowing what each label stands for can make it possible to explore keyword based approaches.
    - the distribution of these labels is extremely skewed. in fact, we can see in the figure below that a considerable part of the labels is used rarely. these labels are very hard to learn, and will result in overfitting. Therefore, unless we manage to get additional data, we can either craft rules based on our understanding of those labels(which is not possible in this case as we don't know what each label stands for), or we can adopt one of the ML algorithms to learn from very few examples, which is out of the scope of this basic approach. Therefore, we will simply remove these labels, as our basic algorithm is unable to model them, and they'll slow down the process.
<br>
<br>
  <p align="center">
    <img width="320" height="200" src="conditional_distribution_labels_counts.png">
  </p>
  <br>

- **On the features :** 
    - Obviously, the most important feature is the URL. Not only are they supposed to be designed to be easily and quickly understandable by humans but they also follow a predefined pattern that makes it possible to interpret them as tabular data, and some parts can be used to define new categorical features. The subdomain and the top level domain are relatively easy to categorize as they often take a value from a well known set of possibilities (always true for the tld, most of the time true for the subdomain which can be customized), so manually defining relevant categories based on the carried meaning and the frequency in the provided training set works quite well. This is not the case for the primary domain or the hostname which are categorized typically through scraping their webpages. In our setting, we want to extract url-related information exclusively from the text of the url. Fortunately, we have a relatively large labeled dataset. Although we don't know the semantic meaning of the labels, we can suppose that hostnames which have the urls they appeared in often tagged with similar labels, are very likely to be in the same category. Based on this intuition, we perform a hdbscan clustering, with "manhattan" distance (good option for our high dimensional labels space) to construct the hostname_category_map which maps the n most frequent hostnames in the training set to their categories (n is specified by the user if the map is to be updated). The original mapping was obtained through considering the 400 most frequently used hostnames. We see in the resulting map that, even though a lot of points were found noisy, the items the identified clusters make a lot of sense, mostly.
    - The text of the url was vectorized using TFIDF. Applying dimensionality reduction techniques such as SVD on the output of the TFIDF vectorizer caused a huge drop in the performance. Of course, the use of sparse representations is necessary to avoid memory problems and improve the compute time.

- **On the model :**
    - As the purpose of this project is to implement a simple and quick solution to this complex problem, we tried only 3 basic types of classifiers : Multioutput classifier (which had the worst performance), OneVsRest classifiers, and classifier chains. 
    - Classifier chains should be more suitable for our problem as they take advantage of the relations between labels, which helps incorporating the information about the likelihood of co-occurences into the model (example : the chances of the subset of labels (e-commerce, clothes, discount) are way higher than the chances of the subset (discount, politics, evolution)
). However, their performance was always below the performance of the OneVsRest classifiers. This is mainly due to the random order of the chain. The performance is expected to improve if we find a convenient order. 
      In both cases, observing the classification report shows that the performance is limited because a huge number of labels was never present in the predictions, which is completely normal as we mentioned earlier that a large set of labels are rare. This highlights the need to a hierarchical modeling approach, which should be the next thing to try. <br>
      Initially, the results were very unsatisfying : (very high precision, very low recall : the binary classifiers fail to learn the labels as they are underrepresented, especially under the binary classification setting)
    - However, we managed to significantly improve the performance by : 

        - Finding good values for the classes weights 
        - Finding a good prediction threshold
        - Making sure to take at least one label per example, and at most about 5 labels : this is based on the observation of the training set. We noticed that all the examples have a number of labels between 1 and 5 (with 87% of the examples have 5 labels).
        However with this modeling approach, many examples get a high number of labels or no labels. To mitigate this effect, we lower the prediction threshold, which ensures that examples will not mostly have fewer labels than the ground truth, and then, if the number of predicted labels is >5, we select the labels that have the 5 highest probabilities ; and if no label was predicted, we add the label with the highest proabability.
        
        With this approach, we managed to get better results and find a good compromise between precision and recall  

</details>



<details>
  <summary>
  
  ## Results :
  
  </summary>

The results after training the model on 80% of the data, and testing on 20% are :

**Label based metrics :** Precision_micro : 55%, recall_micro : 54%

**Sample based metrics :** Hamming Loss : 0.003

Compute time is high, especially during training which is very slow.

</details>

<details>
  <summary>
  
  ## Conclusion :
  
  </summary>

This work is nothing but a first attempt to get a quick grasp of the problem of (Extreme) Multi Label Classification, with severe label imbalance, using a basic approach. It exposes the limitations of traditional multi label classification approaches as soon as the number of possible labels becomes high, in terms of quality of predictions as well as computational time.

To go further in solving this problem, we could consider, for instance, a tree-based hierarchical approach which will be useful in capturing the dependencies between labels and handling the labels distribution imbalance, as well as reducing the training and prediction time significantly. 

 This type of problems has been an active topic for research for a long time, and there are interesting directions to explore, many of which offer open sourced implementations of their solutions.

</details>

## Disclaimer :
The list of ccTLDs ("country-codes-tlds.csv") is from [here](https://gist.github.com/derlin/421d2bb55018a1538271227ff6b1299d#file-country-codes-tlds-csv)