import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sys import argv

## Data Loader Module
# Load data from files
class DataLoader:
    def __init__(self, ratings_file, content_file, targets_file):
        self.ratings_file = ratings_file
        self.content_file = content_file
        self.targets_file = targets_file

    def load_data(self):
        ratings = pd.read_json(self.ratings_file, lines=True)
        content = pd.read_json(self.content_file, lines=True)
        targets = pd.read_csv(self.targets_file)
        return ratings, content, targets
    
## Data Processor Module
# Generate tags from content data and generate similarity matrix from tags
class DataProcessor:
    def __init__(self, content, ratings):
        self.content = content
        self.ratings = ratings

    def filter_box_office(self, box_office):
        if box_office == 'N/A':
            return 'insignificante'

        if type(box_office) == str:
            cleaned_monetary_value = box_office.replace('$', '').replace(',', '')
            int_monetary_value = int(cleaned_monetary_value)
        else:
            int_monetary_value = box_office

        if int_monetary_value < 927821:
            return 'baixo'
        elif int_monetary_value < 17588670:
            return 'medio'
        else:
            return 'alto'
        
    def filter_year(self, year):
        try:
            year = int(year)
        except:
            return ''

        if year < 1900:
            return 'muito_antigo'
        elif year < 1950:
            return 'antigo'
        elif year < 1980:
            return 'quase_moderno'
        elif year < 1990:
            return 'moderno'
        elif year < 2000:
            return 'recente'
        elif year < 2010:
            return 'muito_recente'
        else:
            return 'atual'
        
    def check_oscar(self, string_awards):
        awards = string_awards.split()

        if 'Oscar.' in awards or 'Oscar' in awards:
            return 'Oscar'
        else:
            return ''

    # tags = genre + language + country + actors + year -> textual description of the movie
    def generate_tags(self):
        content_clear = self.content[['ItemId', 'Genre', 'Language', 'Country', 'imdbRating', 'imdbVotes', 'Year', 'Actors', 'Director', 'BoxOffice', 'Awards']].copy()
        content_clear['MovieId'] = content_clear['ItemId']
        content_clear = content_clear.set_index('MovieId')
        content_clear['newId'] = range(0, len(content_clear))

        content_clear['BoxOffice'] = content_clear['BoxOffice'].apply(self.filter_box_office)

        content_clear['Awards'] = content_clear['Awards'].apply(self.check_oscar)

        content_clear['Actors'] = content_clear['Actors'].apply(lambda x: x.replace(" ", ""))	
        content_clear['Actors'] = content_clear['Actors'].apply(lambda x: x.replace(",", " "))

        content_clear['Country'] = content_clear['Country'].apply(lambda x: x.replace(" ", ""))
        content_clear['Country'] = content_clear['Country'].apply(lambda x: x.replace(",", " "))
        
        content_clear['Director'] = content_clear['Director'].apply(lambda x: x.replace(" ", ""))
        content_clear['Director'] = content_clear['Director'].apply(lambda x: x.replace(",", " "))

        content_clear['Year'] = content_clear['Year'].apply(self.filter_year)

        content_clear['tags'] = content_clear['Genre'].str.replace('N/A', '') + ' ' + content_clear['Language'].str.replace('None', '') + ' ' + content_clear['Country'].str.replace('N/A', '') + ' ' + content_clear['Actors'].str.replace('N/A', '') + ' ' + content_clear['Year'] + content_clear['Director'].str.replace('N/A', '') + '' + content_clear['BoxOffice'] + '' + content_clear['Awards']

        content_clear['tags'] = content_clear['tags'].str.replace(',', '')

        self.content_clear = content_clear
        return content_clear

    # TF-IDF matrix to represent the textual description of the movie
    # Similarity between movies based on the TF-IDF matrix -> cosine similarity
    def generate_similarity_matrix(self):
        tfidf = TfidfVectorizer(stop_words='english', max_features=2000)  
        vectorized_data = tfidf.fit_transform(self.content_clear['tags'])
        svd = TruncatedSVD(n_components=100, random_state=42) 
        reduced_content_data = svd.fit_transform(vectorized_data)
        reduced_content_data = reduced_content_data.astype(np.float32)
        similarity_matrix = cosine_similarity(reduced_content_data)
        return similarity_matrix
    
    def map_ids(self):
        item_to_id = {}
        id_to_item = {}
        user_to_id = {}
        id_to_user = {}
        item_ids = []
        user_ids = []
        count = 0
        for item in self.ratings['ItemId']:
            if item not in item_to_id:
                item_to_id[item] = count
                id_to_item[count] = item
                count += 1
            item_ids.append(item_to_id[item])
        count = 0
        for user in self.ratings['UserId']:
            if user not in user_to_id:
                user_to_id[user] = count
                id_to_user[count] = user
                count += 1
            user_ids.append(user_to_id[user])

        return user_to_id, item_to_id, id_to_user, id_to_item, user_ids, item_ids

    def map_content(self):
        item_to_id = {}
        id_to_item = {}
        count = 0
        for item in self.content['ItemId']:
            if item not in item_to_id:
                item_to_id[item] = count
                id_to_item[count] = item
                count += 1
        
        return item_to_id, id_to_item
    
## Recommender System Module
# Funk SVD to predict ratings of movies that already have ratings
# Content-based to predict ratings of movies that don't have ratings
class RecommenderSystem:
    def __init__(self, ratings, content_clear, similarity_matrix, targets, user_to_id, item_to_id, id_to_user, id_to_item, user_ids, item_ids, item_to_id_content):
        self.ratings = ratings
        self.content_clear = content_clear
        self.similarity_matrix = similarity_matrix
        self.targets = targets
        self.user_to_id = user_to_id
        self.item_to_id = item_to_id
        self.id_to_user = id_to_user
        self.id_to_item = id_to_item
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.item_to_id_content = item_to_id_content
        self.user_preferences = None
        self.P = None
        self.Q = None
        self.bu = None
        self.bi = None
        self.mu = None

    # Mini-batch gradient descent to train the model
    def funkSvd(self):
        np.random.seed(42)

        self.ratings['UserId_num'] = self.user_ids
        self.ratings['ItemId_num'] = self.item_ids

        K = 15 # latent factors
        n_users = len(self.user_to_id)
        n_items = len(self.item_to_id)

        P = np.random.normal(scale=1./K, size=(n_users, K))
        Q = np.random.normal(scale=1./K, size=(n_items, K))
        bu = np.zeros(n_users)
        bi = np.zeros(n_items)
        mu = np.mean(self.ratings['Rating'])

        learning_rate = 0.02
        lambda_reg = 0.1 # regularization to avoid overfitting
        epochs = 6 # epochs
        batch_size = 256 

        num_batches = int(np.ceil(len(self.ratings) / float(batch_size)))


        for epoch in range(epochs):
            
            indices = np.arange(len(self.ratings))
            np.random.shuffle(indices)

            for i in range(num_batches):
                # extracting the batch indices
                batch_indices = indices[i * batch_size: (i + 1) * batch_size]
                # extracting the batch from the dataframe
                batch = self.ratings.iloc[batch_indices] 
                users, items, true_ratings = batch['UserId_num'].values, batch['ItemId_num'].values, batch['Rating'].values
                
                preds = mu + bu[users] + bi[items] + np.sum(P[users, :] * Q[items, :], axis=1) # predicted ratings
                error = true_ratings - preds

                # bias update
                bu[users] += learning_rate * (error - lambda_reg * bu[users])
                bi[items] += learning_rate * (error - lambda_reg * bi[items])
                
                # latent factors update
                for u, i, e in zip(users, items, error):
                    P[u, :] += learning_rate * (e * Q[i, :] - lambda_reg * P[u, :])
                    Q[i, :] += learning_rate * (e * P[u, :] - lambda_reg * Q[i, :])
        
        self.P, self.Q, self.bu, self.bi, self.mu = P, Q, bu, bi, mu
    
    def item_id_to_sim_index(self, item_id):
        if item_id in self.content_clear.index:
            return self.content_clear.loc[item_id, 'newId']
        else:
            return None
        
    # Calculate user preferences based on the ratings of the movies that the user has already rated (top 15)
    def calculate_user_preferences(self, unique_users_in_targets):
        user_preferences = {}

        for user_id in unique_users_in_targets:
                # get top 15 rated movies by the user
                user_ratings = self.ratings[self.ratings['UserId'] == user_id].nlargest(15, 'Rating')

                idxs = [self.item_id_to_sim_index(item_id) for item_id in user_ratings['ItemId'].values]

                total_similarity = np.sum(self.similarity_matrix[idxs] * user_ratings['Rating'].values[:, np.newaxis], axis=0)
                total_weights = np.sum(user_ratings['Rating'].values)

                # user preference is the weighted average of the top 15 rated movies
                user_pref = total_similarity / total_weights if total_weights != 0 else np.zeros(self.similarity_matrix.shape[1])

                user_preferences[user_id] = user_pref

        self.user_preferences = user_preferences
    
    # Content-based to predict ratings of movies that don't have ratings
    def content_based(self, item_id_str, user_id_str, mu):
        user_pref = self.user_preferences.get(user_id_str, np.zeros(self.similarity_matrix.shape[1]))
        item_idx = self.item_id_to_sim_index(item_id_str)

        if item_idx is None:
            return mu
        
        top_n = 40 # top 40 similar movies to the movie that we want to predict the rating
        top_indices = np.argsort(-user_pref)[:top_n]

        # Score is the dot product of the user preferences and the similarity matrix
        score = np.dot(user_pref[top_indices], self.similarity_matrix[item_idx, top_indices])

        return 10 * score

    # Predict ratings based on the Funk SVD model and the content-based model
    def predict(self, user_id_str, item_id_str, imdbRatings, imdbVotes):
        if item_id_str not in self.item_to_id:
            calculated_rating = self.content_based(item_id_str, user_id_str, self.mu)
        else:
            user_id = self.user_to_id[user_id_str]
            item_id = self.item_to_id[item_id_str]

            calculated_rating = np.clip(self.mu + self.bu[user_id] + self.bi[item_id] + np.dot(self.P[user_id, :], self.Q[item_id, :]), 1, 10)
        
        imdb_rating_movie = imdbRatings[self.item_to_id_content[item_id_str]]
        imdb_votes_movie = imdbVotes[self.item_to_id_content[item_id_str]]

        # Uses the IMDB rating and the number of votes to calculate the final rating
        return calculated_rating * imdb_rating_movie * imdb_votes_movie
    
    # Re-order the movies based on the predicted ratings
    def order_predictions(self, items_per_user, imdbRatings, imdbVotes):
        self.funkSvd()
        ordered_predictions = {}

        for user, items in items_per_user.items():
            items_with_predictions = [(item, self.predict(user, item, imdbRatings, imdbVotes)) for item in items]

            ordered_items = sorted(items_with_predictions, key=lambda x: x[1], reverse=True)

            ordered_predictions[user] = ordered_items

        return ordered_predictions
    
    def print_result(self, ordered_predictions):
        print('UserId,ItemId')

        for user, items in ordered_predictions.items():
            for item, _ in items:
                print(f'{user},{item}')

def main():
    data_loader = DataLoader(argv[1], argv[2], argv[3])
    ratings, content, targets = data_loader.load_data()

    data_processor = DataProcessor(content, ratings)
    content_clear = data_processor.generate_tags()
    similarity_matrix = data_processor.generate_similarity_matrix()

    user_to_id, item_to_id, id_to_user, id_to_item, user_ids, item_ids = data_processor.map_ids()
    item_to_id_content, id_to_item_content = data_processor.map_content()

    recommender_system = RecommenderSystem(ratings, content_clear, similarity_matrix, targets, user_to_id, item_to_id, id_to_user, id_to_item, user_ids, item_ids, item_to_id_content)

    content['imdbRating'] = pd.to_numeric(content['imdbRating'], errors='coerce').fillna(1)
    content['imdbVotes'] = content['imdbVotes'].str.replace(',', '')
    content['imdbVotes'] = pd.to_numeric(content['imdbVotes'], errors='coerce').fillna(1)
    imdbRatings = content['imdbRating'].to_numpy()
    imdbVotes = content['imdbVotes'].to_numpy()

    unique_users_in_targets = targets['UserId'].unique()
    recommender_system.calculate_user_preferences(unique_users_in_targets)

    items_per_user = targets.groupby('UserId')['ItemId'].apply(list)
    ordered_predictions = recommender_system.order_predictions(items_per_user, imdbRatings, imdbVotes)

    recommender_system.print_result(ordered_predictions)

if __name__ == "__main__":
    main()
