import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import osmnx as ox
import networkx as nx
import folium
from haversine import haversine
from openrouteservice import client
import geopy.distance
from openrouteservice import convert
import requests
import json
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.linear_model import LinearRegression
import re
from nltk.tokenize import word_tokenize
import stellargraph as sg
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.models import Model, Sequential
from keras.layers import Input, Embedding, multiply, Dense, Flatten, Concatenate, Dropout
from keras.optimizers import Adam
import tensorflow as tf
import random
import time
from streamlit_folium import folium_static
import uuid
from streamlit_chat import message, NO_AVATAR
import streamlit as st

# OpenRouteService API 키 설정
api_key = '5b3ce3597851110001cf6248683f2a9afa4343aba7da92fd50c6545e'
clnt = client.Client(key=api_key)

# 데이터 불러오기
df_nodes_essential = pd.read_csv('C:/Users/CAU/Documents/Travel-Route-Recommendation-main/Korea_amenities_essential.csv')
bts_visited_places = pd.read_csv('C:/Users/CAU/Documents/Travel-Route-Recommendation-main/BTS_visited_places.csv')
tripadvisor_review = pd.read_csv('C:/Users/CAU/Documents/Travel-Route-Recommendation-main/tripadvisor_route_review.csv')
tripadvisor_places = pd.read_csv('C:/Users/CAU/Documents/Travel-Route-Recommendation-main/should_visit_places_1.csv')
temp = tripadvisor_places.merge(tripadvisor_review, left_on='ROUTENAME', right_on='ROUTE_ID')

# 데이터 전처리
df_nodes_essential[['x', 'y']] = df_nodes_essential[['x', 'y']].apply(pd.to_numeric, errors='coerce')
df_nodes_essential = df_nodes_essential.dropna(subset=['x', 'y'])
bts_visited_places[['lat', 'lon']] = bts_visited_places[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')

# # Calculate average rating for each route
# average_ratings = tripadvisor_review.groupby('ROUTE_ID')['Rating'].mean()

# # 각 ROUTE_ID에 대한 총 합산 점수 계산
# total_scores = tripadvisor_review.groupby('ROUTE_ID')['Rating'].sum()

# # Define thresholds for recommendation
# # These thresholds are arbitrary for demonstration; adjust them as needed.
# most_recommended_threshold = 4 # Top 25%
# not_recommended_threshold = 3.9  # Bottom 25%

# # Define Mish activation function
# def mish(x):
#     return x * tf.math.tanh(tf.math.softplus(x))

# # Assuming 'temp' is a DataFrame loaded previously
# route_node = temp['ROUTE_ID']
# place_node = temp['Place']
# user_node = temp['USER_ID']

# route_node = route_node.drop_duplicates()
# place_node = place_node.drop_duplicates()
# user_node = user_node.drop_duplicates()

# route_node_ids = pd.DataFrame(route_node)
# place_node_ids = pd.DataFrame(place_node)
# user_node_ids = pd.DataFrame(user_node)

# route_node_ids.set_index('ROUTE_ID', inplace=True)
# place_node_ids.set_index('Place', inplace=True)
# user_node_ids.set_index('USER_ID', inplace=True)

# # Edge data preparation
# user_route_edge = temp[['USER_ID', 'ROUTE_ID']]
# user_route_edge.columns = ['source', 'target']

# route_place_edge = temp[['ROUTE_ID', 'Place']]
# route_place_edge.columns = ['source', 'target']

# start = len(user_route_edge)
# route_place_edge.index = range(start, start + len(route_place_edge))

# g = sg.StellarDiGraph(nodes={'user': user_node_ids, 'route': route_node_ids, 'place': place_node_ids},
#        edges={'user_route': user_route_edge, 'route_place': route_place_edge})

# print(g.info())

# # HIN embedding with Metapath2Vec
# walk_length = 50
# metapaths = [["user", "route", "place", "route", "user"], ["user", "route", "user"]]

# from stellargraph.data import UniformRandomMetaPathWalk

# rw = UniformRandomMetaPathWalk(g)

# walks = rw.run(
#     nodes=list(g.nodes()),  # root nodes
#     length=walk_length,  # maximum length of a random walk
#     n=10,  # number of random walks per root node, repeat count
#     metapaths=metapaths,  # the metapaths
#     seed=42
# )

# from gensim.models import Word2Vec

# str_walks = [[str(n) for n in walk] for walk in walks]
# model = Word2Vec(str_walks, vector_size=128, window=5, min_count=0, sg=1, epochs=5)

# node_ids = model.wv.index_to_key
# x = model.wv.vectors
# y = [g.node_type(node_id) for node_id in node_ids]

# # Embedding vectors
# node_embedding = pd.DataFrame(x, index=node_ids)
# node_embedding['target'] = y

# # Extract user embedding
# User_embedding = node_embedding[node_embedding['target'] == 'user']
# del User_embedding['target']
# User_embedding.index.name = 'USER_ID'

# # Route Embedding
# course_sequence = pd.read_csv('C:/Users/CAU/Documents/Travel-Route-Recommendation-main/should_visit_places_1.csv')
# course_sequence.columns = ["ROUTENAME", "Place", "Latitude", "Longitude"]
# course_sequence_nan = course_sequence[course_sequence['Place'].str.contains("nan", na=True, case=False)]
# course_sequence = course_sequence[course_sequence['Place'].isin(course_sequence_nan['Place']) == False]
# places = course_sequence['Place']

# # Mapping words to index
# word_to_index = {}
# index_to_word = {}
# current_index = 0

# sequences = []
# for place in places:
#     sequence = []
#     for word in place.split(", "):
#         if word not in word_to_index:
#             word_to_index[word] = current_index
#             index_to_word[current_index] = word
#             current_index += 1
#         sequence.append(word_to_index[word])
#     sequences.append(sequence) 

# # Padding sequences
# max_sequence_length = max(len(sequence) for sequence in sequences)
# padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_length)

# embedding_dim = 32
# embedding_output_dim = 64

# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(input_dim=len(word_to_index), output_dim=embedding_dim, input_length=max_sequence_length),
#     tf.keras.layers.LSTM(units=embedding_dim, return_sequences=False),
#     tf.keras.layers.Dense(embedding_output_dim)  # Route embedding output
# ])

# # Embedding
# RNN_embedded_data = model.predict(padded_sequences)
# Route_embedding = pd.DataFrame(RNN_embedded_data, index=course_sequence['ROUTENAME'])
# Route_embedding.index.name = 'ROUTE_ID'

# # Evaluation
# temp = tripadvisor_review.merge(User_embedding, on='USER_ID')
# temp = temp.merge(Route_embedding, on='ROUTE_ID')

# Feature_vec = temp[list(temp.columns[3:])].to_numpy()
# label = temp['Rating'].to_numpy()

# # PCA
# pca = PCA(n_components=128, random_state=150)
# Feature_vec_pca = pca.fit_transform(Feature_vec)
# Feature_vec_pca.shape

# # Define original data
# original_data = Feature_vec_pca
# original_labels = label

# # Train-test split
# training_data, test_data, training_labels, test_labels = train_test_split(Feature_vec_pca, label, test_size=0.4, shuffle=True, random_state=150)

# X = training_data
# y = training_labels  # Label column

# # Parameters
# num_features = X.shape[1]
# num_labels = len(np.unique(y)) + 1
# latent_dim = 128

# # Define the generator with Mish activation function
# def build_generator():
#     model = Sequential()

#     model.add(Dense(128, input_dim=latent_dim, activation=mish))
#     model.add(Dense(256, activation='relu'))
#     model.add(Dense(512, activation='relu'))
#     model.add(Dense(num_features, activation=mish))
#     model.add(Dropout(0.6))

#     noise = Input(shape=(latent_dim,))
#     label = Input(shape=(1,), dtype='int32')
#     label_embedding = Flatten()(Embedding(num_labels, latent_dim)(label))

#     model_input = multiply([noise, label_embedding])
#     output = model(model_input)

#     return Model([noise, label], output)

# # Define the discriminator
# def build_discriminator():
#     img = Input(shape=(num_features,))
#     label = Input(shape=(1,), dtype='int32')
#     label_embedding = Flatten()(Embedding(num_labels, num_features)(label))

#     model_input = Concatenate(axis=1)([img, label_embedding])

#     model = Sequential()
#     model.add(Dense(512, input_dim=num_features + num_features, activation=mish))
#     model.add(Dense(128))
#     model.add(Dense(64))
#     model.add(Dense(1))
#     model.add(Dropout(0.4))

#     validity = model(model_input)

#     return Model([img, label], validity)

# # Build and compile the generator
# generator = build_generator()
# generator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001))

# # Build and compile the discriminator
# discriminator = build_discriminator()
# discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# # Build the combined model
# z = Input(shape=(latent_dim,))
# label = Input(shape=(1,))
# img = generator([z, label])
# discriminator.trainable = False
# valid = discriminator([img, label])

# combined = Model([z, label], valid)
# combined.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001))

# # Train the model
# def train(epochs, batch_size=128):
#     real_labels = np.ones((batch_size, 1))
#     fake_labels = np.zeros((batch_size, 1))

#     for epoch in range(epochs):
#         idx = np.random.randint(0, X.shape[0], batch_size)
#         imgs, labels = X[idx], y[idx]

#         noise = np.random.normal(0, 1, (batch_size, latent_dim))
#         gen_imgs = generator.predict([noise, labels.reshape(-1, 1)])

#         d_loss_real = discriminator.train_on_batch([imgs, labels.reshape(-1, 1)], real_labels)
#         d_loss_fake = discriminator.train_on_batch([gen_imgs, labels.reshape(-1, 1)], fake_labels)
#         d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

#         sampled_labels = np.random.randint(0, num_labels, batch_size).reshape(-1, 1)
#         g_loss = combined.train_on_batch([noise, sampled_labels], real_labels)

#         print(f"{epoch + 1} [D loss: {d_loss[0]:.2f}, accuracy: {100 * d_loss[1]:.2f}] [G loss: {g_loss:.2f}]")

# train(epochs=500)

# def generate_samples(num_samples, labels):
#     noise = np.random.normal(0, 1, (num_samples, latent_dim))
#     gen_data = generator.predict([noise, labels.reshape(-1, 1)])
#     return gen_data

# num_samples_to_generate = 5000
# generated_label = np.array([(i % 5) + 1 for i in range(num_samples_to_generate)])
# generated_data = generate_samples(num_samples=len(generated_label), labels=generated_label)

# # Generate Augmented Data
# augmented_data = np.vstack((original_data, generated_data))
# augmented_labels = np.concatenate((original_labels, generated_label))

# # Data Normalization
# scaler = StandardScaler()
# augmented_data_normalized = scaler.fit_transform(augmented_data)

# # Initialize and Train MLP Regressor
# mlp = MLPRegressor(hidden_layer_sizes=(300, 250, 200, 150, 100, 50), max_iter=500, alpha=1e-5, solver='adam', verbose=1, random_state=150, learning_rate_init=0.01)
# mlp.fit(augmented_data_normalized, augmented_labels)

# # Predict and Evaluate
# predictions = mlp.predict(augmented_data_normalized)
# rmse = mean_squared_error(predictions, augmented_labels)**0.5
# mae = mean_absolute_error(predictions, augmented_labels)

# print(f"RMSE: {rmse}, MAE: {mae}")

# # 샘플 데이터셋에 대한 예측 및 평가
# sample_data_predictions = mlp.predict(scaler.transform(generated_data))
# sample_data_rmse = mean_squared_error(sample_data_predictions, generated_label)**0.5
# sample_data_mae = mean_absolute_error(sample_data_predictions, generated_label)

# # 원본 데이터셋에 대한 예측 및 평가
# original_data_predictions = mlp.predict(scaler.transform(original_data))
# original_data_rmse = mean_squared_error(original_data_predictions, original_labels)**0.5
# original_data_mae = mean_absolute_error(original_data_predictions, original_labels)

# # 결과 출력
# print(f"Original Dataset RMSE: {original_data_rmse}, MAE: {original_data_mae}")
# print(f"Sample Dataset RMSE: {sample_data_rmse}, MAE: {sample_data_mae}")
# print(f"Augmented Dataset RMSE: {rmse}, MAE: {mae}")

# # 데이터 정규화
# scaler = StandardScaler()
# training_data_normalized = scaler.fit_transform(training_data)
# test_data_normalized = scaler.transform(test_data)

# # MLP Regressor 초기화 및 훈련
# mlp = MLPRegressor(hidden_layer_sizes=(300, 250, 200, 150, 100, 50), max_iter=500, alpha=1e-5, solver='adam', verbose=0, random_state=150, learning_rate_init=0.01)
# mlp.fit(training_data_normalized, training_labels)

# # 테스트 데이터에 대한 예측
# mlp_pred = mlp.predict(test_data_normalized)

# # RMSE와 MAE 계산
# rmse = mean_squared_error(mlp_pred, test_labels)**0.5
# mae = mean_absolute_error(mlp_pred, test_labels)

# print(rmse)
# print(mae)

# #Data for plotting
# errors = ['rmse', 'mae']
# values = [rmse, mae]

# # 결과 시각화
# plt.figure(figsize=(8,5))
# plt.bar(['RMSE', 'MAE'], [rmse, mae], color=['blue', 'orange'])
# plt.xlabel('Error Type')
# plt.ylabel('Value')
# plt.title('RMSE and MAE Visualization')
# plt.ylim(0, max([rmse, mae]) + 0.1 * max([rmse, mae]))
# plt.show()

# # Identify Most and Not Recommended Routes
# most_recommended_routes = average_ratings[average_ratings >= most_recommended_threshold].index.tolist()
# not_recommended_routes = average_ratings[average_ratings <= not_recommended_threshold].index.tolist()

# print("Most Recommended Routes: \n", most_recommended_routes)
# print("Not Recommended Routes: \n", not_recommended_routes)

# # 'most_recommended_routes'와 'not_recommended_routes' 리스트를 데이터프레임으로 변환
# most_recommended_df = pd.DataFrame(most_recommended_routes, columns=['ROUTE_ID'])
# most_recommended_df['Recommendation'] = 'Most Recommended'

# not_recommended_df = pd.DataFrame(not_recommended_routes, columns=['ROUTE_ID'])
# not_recommended_df['Recommendation'] = 'Not Recommended'

# # Most Recommended와 Not Recommended 데이터프레임에 합산 점수 추가
# most_recommended_df['Total_Score'] = most_recommended_df['ROUTE_ID'].map(total_scores)
# not_recommended_df['Total_Score'] = not_recommended_df['ROUTE_ID'].map(total_scores)

# # Most Recommended와 Not Recommended 데이터프레임에 평균 평점 추가
# most_recommended_df['Average_Rating'] = most_recommended_df['ROUTE_ID'].map(average_ratings)
# not_recommended_df['Average_Rating'] = not_recommended_df['ROUTE_ID'].map(average_ratings)

# # 두 데이터프레임을 하나로 합치기
# combined_df_with_scores = pd.concat([most_recommended_df, not_recommended_df], ignore_index=True)

# # CSV 파일로 저장
# combined_df_with_scores.to_csv('C:/Users/CAU/Documents/Travel-Route-Recommendation-main/recommended_routes_with_scores.csv', index=False)

# print("CSV 파일이 성공적으로 저장되었습니다.")

# # 단계 1: 'Most Recommended' 경로 추출
# combined_df_with_scores = pd.read_csv('C:/Users/CAU\Documents/Travel-Route-Recommendation-main/recommended_routes_with_scores.csv')
# most_recommended_routes = combined_df_with_scores[combined_df_with_scores['Recommendation'] == 'Most Recommended']

# # 단계 2: 경로에 해당하는 'Place' 찾기
# route_list = pd.read_csv('C:/Users/CAU/Documents/Travel-Route-Recommendation-main/tripadvisor_route_list.csv')
# recommended_places = route_list[route_list['ROUTENAME'].isin(most_recommended_routes['ROUTE_ID'])]

# # 단계 3: 무작위로 'Place' 추천
# places_to_recommend = recommended_places['Place'].unique()
# number_of_places_to_recommend = random.randint(4, 8) # 4에서 8개 사이의 숫자를 무작위로 선택
# recommended_places_list = random.sample(list(places_to_recommend), number_of_places_to_recommend)

# # 추천된 장소를 데이터프레임으로 변환하고 CSV 파일로 저장
# recommended_places_df = pd.DataFrame(recommended_places_list, columns=['Recommended Places'])
# recommended_places_df.to_csv('C:/Users/CAU/Documents/Travel-Route-Recommendation-main/recommended_places.csv', index=False)

# print("추천 장소: ", recommended_places_list)
# print("CSV 파일이 'C:/Users/CAU/Documents/Travel-Route-Recommendation-main/recommended_places.csv'로 저장되었습니다.")

# recommended_places.csv 파일 읽기
places_df = pd.read_csv('C:/Users/CAU/Documents/Travel-Route-Recommendation-main/recommended_places.csv')

#좌표 데이터를 이용하여 BallTree 객체 생성
tree = BallTree(np.deg2rad(df_nodes_essential[['y', 'x']].values), metric='haversine')

def search_nearby_places(lat, lon, tomtom_key, radius, retries=3, delay=1):
    categories = "7311,7315,9376,9377,9378,9379,7380,7381"  # 레스토랑, 호텔, 모텔, 화장실, 버스 정류장, 지하철 역, 카페의 카테고리 ID
    url = f"https://kr-api.tomtom.com/search/2/nearbySearch/.json?lat={lat}&lon={lon}&radius={radius}&categorySet={categories}&key={tomtom_key}&language=ko-KR"
    for attempt in range(retries):
        response = requests.get(url)
        if response.status_code == 200:
            try:
                data = response.json()
                results = data.get('results', [])
                if not results:
                    st.write("No results found. Please check the category IDs and API response.")
                return results
            except json.JSONDecodeError:
                print("Error: Failed to decode JSON response")
                print(f"Response content: {response.text}")
                return []
        elif response.status_code == 429:
            print(f"Error: Received status code 429 from TomTom API. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
        else:
            print(f"Error: Received status code {response.status_code} from TomTom API")
            st.write(f"Response content: {response.text}")
            return []

    st.write("Error: Exceeded maximum retries for TomTom API")
    return []

def filter_places_by_name(input_name, places_df): 
    # Filter places that start or end with the input name 
    filtered_places = places_df[places_df['name'].str.startswith(input_name) | places_df['name'].str.endswith(input_name)] 
    return filtered_places

def get_user_input_and_recommend(form_key): 
    with st.form(key=form_key):
        user_input = st.text_input("출발지, 목적지, 테마, 방문할 장소의 수를 입력하세요 (예: 서울역, 강남역, bts, 6): ")
        radius = st.slider("검색 반경 (km)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        submit_button = st.form_submit_button(label='Submit')
        
        if submit_button:
            try:
                start, destination, theme, num_visit = user_input.split(", ") 
            except ValueError:
                st.write("입력 형식이 잘못되었습니다. 다시 입력해주세요.")
                return None

            # Filter and select start point
            filtered_start_places = filter_places_by_name(start, df_nodes_essential)
            if filtered_start_places.empty:
                st.write("해당 출발지가 없습니다. 다시 입력해주세요.")
                return None
            st.write("출발지 후보:")
            for idx, row in filtered_start_places.iterrows():
                st.write(f"{idx}: {row['name']}")

            start_idx = st.text_input("출발지 번호를 입력하신 후 Enter키를 눌러주세요: ")
            if start_idx:
                try:
                    start_idx = int(start_idx)
                    if start_idx not in filtered_start_places.index:
                        st.write("유효하지 않은 출발지 번호입니다. 다시 입력해주세요.")
                        return None
                    start_point = filtered_start_places.loc[start_idx][['y', 'x']].values
                except ValueError:
                    st.write("유효하지 않은 입력입니다. 숫자를 입력해주세요.")
                    return None

            # Filter and select destination point
            filtered_destination_places = filter_places_by_name(destination, df_nodes_essential)
            if filtered_destination_places.empty:
                st.write("해당 목적지가 없습니다. 다시 입력해주세요.")
                return None
            st.write("목적지 후보:")
            for idx, row in filtered_destination_places.iterrows():
                st.write(f"{idx}: {row['name']}")

            destination_idx = st.text_input("목적지 번호를 입력하신 후 Enter키를 눌러주세요: ")
            if destination_idx:
                try:
                    destination_idx = int(destination_idx)
                    if destination_idx not in filtered_destination_places.index:
                        st.write("유효하지 않은 목적지 번호입니다. 다시 입력해주세요.")
                        return None
                    destination_point = filtered_destination_places.loc[destination_idx][['y', 'x']].values
                except ValueError:
                    st.write("유효하지 않은 입력입니다. 숫자를 입력해주세요.")
                    return None

            if theme.lower() not in ['trip', ' ']:
                st.write("해당 테마가 없습니다. 다시 입력해주세요.")
                return None

            try:
                num_visit = int(num_visit)
                if num_visit < 2 or num_visit > 10:
                    st.write("방문할 장소의 수는 2~10이어야 합니다.")
                    return None
            except ValueError:
                st.write("방문할 장소의 수는 숫자로 입력해주세요.")
                return None

            if 'start_point' not in locals() or 'destination_point' not in locals():
                return None

            routes = []

            if theme.lower() == "trip":
                # 출발지와 목적지 사이의 거리 계산
                distance_limit = haversine(start_point, destination_point)
                # 거리 내에 있는 장소만 선택
                tripadvisor_places['distance_from_start'] = tripadvisor_places.apply(lambda row: haversine(start_point, [row['Latitude'], row['Longitude']]), axis=1)
                tripadvisor_places['distance_from_destination'] = tripadvisor_places.apply(lambda row: haversine(destination_point, [row['Latitude'], row['Longitude']]), axis=1)
                filtered_places = tripadvisor_places[(tripadvisor_places['distance_from_start'] <= distance_limit) & (tripadvisor_places['distance_from_destination'] <= distance_limit)]

                # 중복된 장소 제거
                filtered_places = filtered_places.drop_duplicates(subset=['Place'])

                # 테마 장소 수 검사 및 조정
                num_visit = min(len(filtered_places), num_visit)
                if len(filtered_places) < num_visit:
                    st.write("테마 장소가 부족합니다. 표시할 수 있는 최대 테마 장소는 {}개입니다.".format(len(filtered_places)))

                # 출발지와의 거리를 기준으로 정렬
                ordered_places = filtered_places.sort_values('distance_from_start')
                visit_points = ordered_places.head(num_visit)

                # Ensure unique places in visit_points
                visit_points = visit_points.drop_duplicates(subset=['Place'])

                # Create routes and search for nearby places
                tomtom_key = "GqGLjbuMGbhvTGRIOUf0G4ZStV9AU97j"

                # 출발지 주변의 장소 찾기
                nearby_places_start = search_nearby_places(start_point[0], start_point[1], tomtom_key, radius=radius*1000)
                st.write(f"출발지 {start} 주변의 장소:")
                for place in nearby_places_start:
                    st.write(f"- {place['poi']['name']} (위도: {place['position']['lat']}, 경도: {place['position']['lon']})")

                for i in range(num_visit):
                    routes.append([start_point, visit_points.iloc[i][['Latitude', 'Longitude']].values.tolist(), destination_point])
                    lat, lon = visit_points.iloc[i][['Latitude', 'Longitude']].values.tolist()
                    nearby_places = search_nearby_places(lat, lon, tomtom_key, radius=radius*1000)
                    st.write(f"경유지 {i+1} 주변의 장소:")
                    for place in nearby_places:
                        st.write(f"- {place['poi']['name']} (위도: {place['position']['lat']}, 경도: {place['position']['lon']})")

                # 목적지 주변의 장소 찾기
                nearby_places_destination = search_nearby_places(destination_point[0], destination_point[1], tomtom_key, radius=radius*1000)
                st.write(f"목적지 {destination} 주변의 장소:")
                for place in nearby_places_destination:
                    st.write(f"- {place['poi']['name']} (위도: {place['position']['lat']}, 경도: {place['position']['lon']})")

                st.write(f"{num_visit}개의 방문 추천 장소는 다음과 같습니다.")
                for i, row in visit_points.iterrows():
                    st.write(f"{row['Place']} (위도: {row['Latitude']}, 경도: {row['Longitude']})")

                route_segments = []
                route_segments.append(start)
                for i in range(num_visit):
                    route_segments.append(visit_points.iloc[i]['Place'])
                route_segments.append(destination)
                complete_route = " -> ".join(route_segments)
                st.write(f"전체 경로: {complete_route}")

            return start_point, destination_point, visit_points[['Latitude', 'Longitude']].values
        return None

def generate_route_with_chatbot(start_point, destination_point, waypoints): 
    m = folium.Map(location=start_point, zoom_start=14)
    # 출발지, 경유지, 목적지를 OpenRouteService가 사용하는 형식으로 변환 
    start_point_ors = [start_point[1], start_point[0]] 
    destination_point_ors = [destination_point[1], destination_point[0]] 
    waypoints_ors = [[point[1], point[0]] for point in waypoints]

    # TomTom API를 사용하여 실시간 교통 정보 가져오기
    tomtom_key = "GqGLjbuMGbhvTGRIOUf0G4ZStV9AU97j"
    # TomTom Traffic API for real-time traffic information
    traffic_url = f"https://kr-api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={start_point[0]},{start_point[1]}&key={tomtom_key}"
    response_traffic = requests.get(traffic_url)
    traffic_data = json.loads(response_traffic.text)
    folium.Marker(location=[start_point[0], start_point[1]], popup='Traffic Info', icon=folium.Icon(color='red')).add_to(m)

    route_url = f"https://kr-api.tomtom.com/routing/1/calculateRoute/{start_point[0]},{start_point[1]}:{destination_point[0]},{destination_point[1]}/json?computeBestOrder=true&key={tomtom_key}"
    response = requests.get(route_url)
    route_data = json.loads(response.text)

    # 경로의 좌표 추출
    points = route_data['routes'][0]['legs'][0]['points']
    coordinates = [(point['latitude'], point['longitude']) for point in points]
    # Extract route coordinates from the API response
    coordinates = [(point['latitude'], point['longitude']) for point in route_data['routes'][0]['legs'][0]['points']]

    # 거리 행렬 계산
    locations = [start_point_ors] + waypoints_ors + [destination_point_ors]
    matrix = clnt.distance_matrix(locations)

    # 지도 생성
    m = folium.Map(location=start_point, zoom_start=14)
    folium.Marker(location=[start_point[0], start_point[1]], popup='출발지', icon=folium.Icon(color='green')).add_to(m)

    # 방문 순서대로 번호 매기기
    for i, waypoint in enumerate(waypoints):
        folium.Marker(
            location=[waypoint[0], waypoint[1]],
            popup=f"방문 순서: {i + 1}",
            icon=folium.Icon(color='purple', number=i + 1),
        ).add_to(m)

        # 출발지 주변의 장소 찾기
        nearby_places_start = search_nearby_places(start_point[0], start_point[1], tomtom_key, radius=1000)  # Add radius argument
        for place in nearby_places_start:
            folium.Marker(
                location=[place['position']['lat'], place['position']['lon']],
                popup=f"{place['poi']['name']}",
                icon=folium.Icon(color='brown', icon='info-sign'),
            ).add_to(m)

        # 경유지 주변의 장소를 검색하고 지도에 표시
        nearby_places = search_nearby_places(waypoint[0], waypoint[1], tomtom_key, radius=1000)  # Add radius argument
        for place in nearby_places:
            folium.Marker(
                location=[place['position']['lat'], place['position']['lon']],
                popup=f"{place['poi']['name']}",
                icon=folium.Icon(color='green', icon='info-sign'),
            ).add_to(m)

    # 목적지 주변의 장소 찾기
    nearby_places_destination = search_nearby_places(destination_point[0], destination_point[1], tomtom_key, radius=1000)  # Add radius argument
    for place in nearby_places_destination:
        folium.Marker(
            location=[place['position']['lat'], place['position']['lon']],
            popup=f"{place['poi']['name']}",
            icon=folium.Icon(color='purple', icon='info-sign'),
        ).add_to(m)

    folium.Marker(location=[destination_point[0], destination_point[1]], popup='목적지', icon=folium.Icon(color='red')).add_to(m)

    # 대체 경로 갯수 설정
    alternative_routes_num = 3
    # 대체 경로를 저장할 리스트
    alternative_routes = []

    # 색상 리스트 설정
    colors = ['red', 'black', 'brown', 'purple', 'darkorange']

    total_distance = 0  # Add this line

    for i in range(alternative_routes_num):
        # 경유지에 랜덤한 좌표를 추가
        waypoints_ors_altered = waypoints_ors + [[waypoints_ors[-1][0] + (i + 1) * 0.001, waypoints_ors[-1][1] + (i + 1) * 0.001]]
        # 좌표 설정
        coordinates_altered = [start_point_ors] + waypoints_ors_altered + [destination_point_ors]
        # 경로 계산
        route = clnt.directions(coordinates=coordinates_altered, profile='foot-walking', units='km', optimize_waypoints=True, format='geojson')

        # 경로 계산이 성공한 경우에만 경로를 대체 경로 리스트에 추가
        if 'features' in route:
            alternative_routes.append(route)
            total_distance += route['features'][0]['properties']['summary']['distance']

    # 각 대체 경로를 지도에 그리기
    for i, route in enumerate(alternative_routes):
        for feature in route['features']:
            folium.PolyLine(locations=[list(reversed(coord)) for coord in feature['geometry']['coordinates']], color=colors[i]).add_to(m)

    # OpenRouteService를 이용한 경로 계산
    #[“driving-car”, “driving-hgv”, “foot-walking”, “foot-hiking”, “cycling-regular”, “cycling-road”,”cycling-mountain”, “cycling-electric”,]
    coordinates = [start_point_ors] + waypoints_ors + [destination_point_ors]
    route = clnt.directions(coordinates=coordinates, profile='foot-walking', units='km', optimize_waypoints=True, format='geojson')

    # 경로 그리기
    for feature in route['features']:
        folium.PolyLine(locations=[list(reversed(coord)) for coord in feature['geometry']['coordinates']], color='blue').add_to(m)

    # 전체 경로 그리기
    full_route_coordinates = [start_point_ors] + waypoints_ors + [destination_point_ors]
    full_route = clnt.directions(coordinates=full_route_coordinates, profile='foot-walking', units='km', optimize_waypoints=True, format='geojson')
    for feature in full_route['features']:
        folium.PolyLine(locations=[list(reversed(coord)) for coord in feature['geometry']['coordinates']], color='green').add_to(m)

    # Draw the route on the map
    folium.PolyLine(locations=coordinates, color='blue').add_to(m)

    # Add markers for start, waypoints, and destination
    folium.Marker(location=start_point, popup='출발지', icon=folium.Icon(color='green')).add_to(m)
    for i, waypoint in enumerate(waypoints):
        folium.Marker(location=waypoint, popup=f"방문 순서: {i + 1}", icon=folium.Icon(color='purple', number=i + 1)).add_to(m)
    folium.Marker(location=destination_point, popup='목적지', icon=folium.Icon(color='red')).add_to(m)

    # Extract the distance and duration from the route data
    distance = route['features'][0]['properties']['summary']['distance']
    duration = route['features'][0]['properties']['summary']['duration']

    # Convert the distance from meters to kilometers
    distance_km = distance / 1000

    # Calculate the duration based on a walking speed of 5km/h
    duration_hours = distance_km / 3.5

    # Convert the duration from seconds to hours and minutes
    hours, remainder = divmod(duration, 3600)
    minutes, _ = divmod(remainder, 60)

    # Print the total distance and duration
    st.write(f"전체 거리: {total_distance/5} km")
    st.write(f"예상 소요 시간: {int(hours)}시간 {int(minutes)}분")

    return m

# Main loop
form_counter = 0
continue_loop = True

while continue_loop: 
    form_key = f"user_input_form_{form_counter}"
    user_input = get_user_input_and_recommend(form_key)
    if user_input is None:
        form_counter += 1
        print("Invalid input. Please try again.")
        break  # or return, depending on your loop structure

    start_point, destination_point, num_visit = user_input

    route_map = generate_route_with_chatbot(start_point, destination_point, num_visit)

    folium_static(route_map)

    choice = st.text_input("계속해서 경로를 생성하시겠습니까? (y/n): ")
    if choice.lower() != 'y':
        continue_loop = False
    form_counter += 1
    