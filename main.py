# Data Structures
import numpy  as np
import pandas as pd
import geopandas as gpd
import json

# Corpus Processing
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import nltk.corpus
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from nltk import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# K-Means
from sklearn import cluster

# Visualization and Analysis
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import silhouette_samples, silhouette_score
from wordcloud import WordCloud

# Map Viz
import folium
#import branca.colormap as cm
from branca.element import Figure
data = pd.read_csv('C:/Git_Repo/national_anthem_classification/input_dataset/anthems_kaggle.csv', encoding='utf-8') #reading the dataset
data.columns = map(str.lower, data.columns)
data.head()
data.info()
data[data['alpha-2'].isna()]
data['alpha-2'].iloc[168] = "NA"  # link https://www.iban.com/country-codes
print(data[data['alpha-2'].isna()])
corpus_in_begining = data['anthem'].tolist()  # making a list of all anthems of 190 countries
len(corpus_in_begining)
print(corpus_in_begining[97][0:]) #india's national anthem in english 
print("-------------------")
print(corpus_in_begining[18][0:447]) #Hungary till 447th letter as it is long anthem
# removes a list of words (ie. stopwords) from a tokenized list.
def remove_words(list_of_tokens, list_of_words):
    return [token for token in list_of_tokens if token not in list_of_words]

# applies stemming to a list of tokenized words
def apply_stemming(list_of_tokens, stemmer):
    return [stemmer.stem(token) for token in list_of_tokens]

# removes any words composed of less than 2 or more than 21 letters
def two_or_22_letters(list_of_tokens):
    two_or_22_letter_word = []
    for token in list_of_tokens:
        if len(token) <= 2 or len(token) >= 21:
            two_or_22_letter_word.append(token)
    return two_or_22_letter_word
def prepare_corpus(corpus, language):   
    stopwords = nltk.corpus.stopwords.words(language)
    param_stemmer = SnowballStemmer(language)
    countries_list = [line.rstrip('\n') for line in open('C:/Git_Repo/national_anthem_classification/input_dataset/countries.txt')] # Load .txt file line by line
    nationalities_list = [line.rstrip('\n') for line in open('C:/Git_Repo/national_anthem_classification/input_dataset/nationalities.txt')] # Load .txt file line by line
    other_words = [line.rstrip('\n') for line in open('C:/Git_Repo/national_anthem_classification/input_dataset/stopwords_scrapmaker.txt')] # Load .txt file line by line
    
    for single_doc in corpus:
        index = corpus.index(single_doc)
        corpus[index] = corpus[index].replace(u'\ufffd', '8')   # Replaces the ASCII '�' symbol with '8'
        corpus[index] = corpus[index].replace(',', '')          # Removes commas
        corpus[index] = corpus[index].rstrip('\n')              # Removes line breaks
        corpus[index] = corpus[index].casefold()                # Makes all letters lowercase
        
        corpus[index] = re.sub('\W_',' ', corpus[index])        # removes specials characters and leaves only words
        corpus[index] = re.sub("\S*\d\S*"," ", corpus[index])   # removes numbers and words concatenated with numbers IE h4ck3r. Removes road names such as BR-381.
        corpus[index] = re.sub("\S*@\S*\s?"," ", corpus[index]) # removes emails and mentions (words with @)
        corpus[index] = re.sub(r'http\S+', '', corpus[index])   # removes URLs with http
        corpus[index] = re.sub(r'www\S+', '', corpus[index])    # removes URLs with www

        list_of_tokens = word_tokenize(corpus[index])
        two_or_22_letter_word = two_or_22_letters(list_of_tokens)

        list_of_tokens = remove_words(list_of_tokens, stopwords)
        list_of_tokens = remove_words(list_of_tokens, two_or_22_letter_word)
        list_of_tokens = remove_words(list_of_tokens, countries_list)
        list_of_tokens = remove_words(list_of_tokens, nationalities_list)
        list_of_tokens = remove_words(list_of_tokens, other_words)
        
        list_of_tokens = apply_stemming(list_of_tokens, param_stemmer)
        list_of_tokens = remove_words(list_of_tokens, other_words)

        corpus[index]   = " ".join(list_of_tokens)
        corpus[index] = unidecode(corpus[index])
    else:
        print("last list_of_tokens in corpus",list_of_tokens)
        print("last corpus[index] in corpus",corpus[index])
        print("last two_or_22_letter_word in corpus",two_or_22_letter_word)

    return corpus
language = 'english'
corpus = prepare_corpus(corpus_in_begining.copy(), language)
print(corpus_in_begining[18][0:460]) 
print("==================")
print(corpus[18][0:460]) #corpus[country][anthem]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus) ## assign score to each word in each list
tf_idf = pd.DataFrame(data = X.toarray(), columns=vectorizer.get_feature_names_out()) ## get_feature_names will return the name of word

final_df = tf_idf

print("{} rows".format(final_df.shape))
final_df.T.nlargest(7, 0)
# kmeans function
def run_KMeans(max_k, data): #max_k number of clusters
    max_k += 1
    kmeans_results = dict()
    distortions = []
    for k in range(2 , max_k): #min 2 clusters
        kmeans = cluster.KMeans(n_clusters = k
                               , init = 'k-means++'
                               , n_init = 10
                               , tol = 0.0001
                               , random_state = 1
                               , algorithm = 'full')

        kmeans_results.update( {k : kmeans.fit(data)} ) #dictionary of K means model object

    return kmeans_results
def printAvg(avg_dict):
    for avg in sorted(avg_dict.keys(), reverse=True):
        print("Avg: {}\tK:{}".format(avg.round(4), avg_dict[avg]))
        
def plotSilhouette(df, n_clusters, kmeans_labels, silhouette_avg):
    fig, ax1 = plt.subplots(1)
    fig.set_size_inches(8, 6)
    ax1.set_xlim([-0.2, 1])
    ax1.set_ylim([0, len(df) + (n_clusters + 1) * 10])
    
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--") # The vertical line for average silhouette score of all the values
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.title(("Silhouette analysis for K = %d" % n_clusters), fontsize=10, fontweight='bold')
    
    y_lower = 10
    sample_silhouette_values = silhouette_samples(df, kmeans_labels) # Compute the silhouette scores for each sample
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[kmeans_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i)) # Label the silhouette plots with their cluster numbers at the middle
        y_lower = y_upper + 10  # Compute the new y_lower for next plot. 10 for the 0 samples
    plt.show()
    
        
def silhouette(kmeans_dict, df, plot=False):
    df = df.to_numpy()
    avg_dict = dict()
    for n_clusters, kmeans in kmeans_dict.items():      
        kmeans_labels = kmeans.predict(df)
        silhouette_avg = silhouette_score(df, kmeans_labels) # Average Score for all Samples
        # print(silhouette_avg)
        avg_dict.update( {silhouette_avg : n_clusters} )
    
        if(plot): plotSilhouette(df, n_clusters, kmeans_labels, silhouette_avg)
    return avg_dict
# avg_dict = silhouette(kmeans_results, final_df)
# print(avg_dict)
k = 8 # test for 8 clusters
kmeans_results = run_KMeans(k, final_df)
def get_top_features_cluster(tf_idf_array, prediction, n_feats):
    labels = np.unique(prediction)
    dfs = []
    for label in labels:
        id_temp = np.where(prediction==label) # indices for each cluster
        x_means = np.mean(tf_idf_array[id_temp], axis = 0) # returns average score across cluster
        sorted_means = np.argsort(x_means)[::-1][:n_feats] # indices with top 20 scores
        features = vectorizer.get_feature_names_out()
        best_features = [(features[i], x_means[i]) for i in sorted_means]
        df = pd.DataFrame(best_features, columns = ['features', 'score'])
        dfs.append(df)
    return dfs

def plotWords(dfs, n_feats):
    plt.figure(figsize=(8, 4))
    for i in range(0, len(dfs)):
        plt.title(("Most Common Words in Cluster {}".format(i)), fontsize=10, fontweight='bold')
        sns.barplot(x = 'score' , y = 'features', orient = 'h' , data = dfs[i][:n_feats])
        plt.show()
best_result = 5
kmeans = kmeans_results.get(best_result)
# Model Prdecition
final_df_array = final_df.to_numpy()
prediction = kmeans.predict(final_df)
print(final_df)
prediction
n_features = 20
dfs = get_top_features_cluster(final_df_array, prediction, n_features)
plotWords(dfs, 13)
common_words=[]
for i in dfs:
  common_words.append(list(i['features'].values)) # to store common words along with the label in final csv.
  # Transforms a centroids dataframe into a dictionary to be used on a WordCloud.
def centroidsDict(centroids, index):
    a = centroids.T[index].sort_values(ascending = False).reset_index().values
    centroid_dict = dict()

    for i in range(0, len(a)):
        centroid_dict.update( {a[i,0] : a[i,1]} )

    return centroid_dict

def generateWordClouds(centroids):
    wordcloud = WordCloud(max_font_size=100, background_color = 'white')
    for i in range(0, len(centroids)):
        centroid_dict = centroidsDict(centroids, i)        
        wordcloud.generate_from_frequencies(centroid_dict)

        plt.figure()
        plt.title('Cluster {}'.format(i))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()
centroids = pd.DataFrame(kmeans.cluster_centers_)
centroids.columns = final_df.columns
generateWordClouds(centroids)
# Assigning the cluster labels to each country
labels = kmeans.labels_ 
data['label'] = labels
data['common_words'] = data['label'].map({0:common_words[0],1:common_words[1],2:common_words[2],3:common_words[3],4:common_words[4]})
data['label_name'] = data['label'].map({0:"Praise the lord cluster",1:" Fatherly Patriot Cluster",2:"Motherly Nature Cluster",3:"Patriot Cluster",4:" Mighty Military Cluster"})
data.to_csv("./grouped_national_anthems.csv",index=False)
data.head()
import json
import geopandas as gpd

# Loading countries polygons
geo_path = '/kaggle/input/text-files-related-to-national-anthem-clustering/world-countries.json'
country_geo = json.load(open(geo_path))
gpf = gpd.read_file(geo_path)

# Merging on the alpha-3 country codes
merge = pd.merge(gpf, data, left_on='id', right_on='alpha-3')
data_to_plot = merge[["id", "name", "label","label_name", "geometry"]]

data_to_plot.head(5)
import branca.colormap as cm

# Creating a discrete color map
values = data_to_plot[['label']].to_numpy()
color_step = cm.StepColormap(['r', 'y','g','b', 'm'], vmin=values.min(), vmax=values.max(), caption='step')

color_step
import folium
from branca.element import Figure

def make_geojson_choropleth(display, data, colors):
    '''creates geojson choropleth map using a colormap, with tooltip for country names and groups'''
    group_dict = data.set_index('id')['label'] # Dictionary of Countries IDs and Clusters
    tooltip = folium.features.GeoJsonTooltip(["name", "label_name"], aliases=display, labels=True)
    return folium.GeoJson(data[["id", "name","label_name","geometry"]],
                          style_function = lambda feature: {
                               'fillColor': colors(group_dict[feature['properties']['id']]),
                               #'fillColor': test(feature),
                               'color':'black',
                               'weight':0.5
                               },
                          highlight_function = lambda x: {'weight':2, 'color':'black'},
                          smooth_factor=2.0,
                          tooltip = tooltip)

# Makes map appear inline on notebook
def display(m, width, height):
    """Takes a folium instance and embed HTML."""
    fig = Figure(width=width, height=height)
    fig.add_child(m)
    #return fig
    # Initializing our Folium Map
m = folium.Map(location=[43.5775, -10.106111], zoom_start=2.3, tiles='cartodbpositron')

# Making a choropleth map with geojson
geojson_choropleth = make_geojson_choropleth(["Country:", "Group:"], data_to_plot, color_step)
geojson_choropleth.add_to(m)

width, height = 1300, 675
display(m, width, height)
m
