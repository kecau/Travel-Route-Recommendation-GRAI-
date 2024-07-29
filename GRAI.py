import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import osmnx as ox
import networkx as nx
import folium
from haversine import haversine
from openrouteservice import client
import geopy.distance
from openrouteservice import convert #distance_matrix, convert
import requests
import json
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.linear_model import LinearRegression
import re
from nltk.tokenize import word_tokenize
import time

# OpenRouteService API 키 설정
api_key = '5b3ce3597851110001cf6248683f2a9afa4343aba7da92fd50c6545e'  # 여기에 OpenRouteService API 키를 입력하세요
clnt = client.Client(key=api_key)

# 데이터 불러오기
df_nodes_essential = pd.read_csv('C:/Users/CAU/Documents/Travel-Route-Recommendation-main/Seoul_Gyeonggi_amenities_essential.csv')
bts_visited_places = pd.read_csv('C:/Users/CAU/Documents/Travel-Route-Recommendation-main/BTS_visited_places.csv')

# 데이터 전처리
df_nodes_essential[['x', 'y']] = df_nodes_essential[['x', 'y']].apply(pd.to_numeric, errors='coerce')
df_nodes_essential = df_nodes_essential.dropna(subset=['x', 'y'])
bts_visited_places[['lat', 'lon']] = bts_visited_places[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')

# NaN 값을 'unknown'으로 대체하고 모든 값을 문자열로 변환
df_nodes_essential['name'] = df_nodes_essential['name'].fillna('unknown').astype(str)

# 장소명을 벡터 형태로 변환
word2vec = Word2Vec(df_nodes_essential['name'].apply(lambda x: [x]), min_count=1)
df_nodes_essential['name_vec'] = df_nodes_essential['name'].apply(lambda x: word2vec.wv[x])

# 학습 데이터와 테스트 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(df_nodes_essential['name_vec'], df_nodes_essential[['x', 'y']], test_size=0.2, random_state=42)

# 모델 학습
model = LinearRegression()
model.fit(list(X_train), y_train)

# 모델 평가
print("Training score: ", model.score(list(X_train), y_train))
print("Testing score: ", model.score(list(X_test), y_test))

# 좌표 데이터를 이용하여 BallTree 객체 생성
tree = BallTree(np.deg2rad(df_nodes_essential[['y', 'x']].values), metric='haversine')

def search_nearby_places(lat, lon, tomtom_key, retries=3, delay=1):
    categories = "7311,7315,9376,9377,9378,9379,7380,7381"  # 레스토랑, 호텔, 모텔, 화장실, 버스 정류장, 지하철 역, 카페의 카테고리 ID
    radius = 10000  # 검색 반경을 10km로 설정
    url = f"https://kr-api.tomtom.com/search/2/nearbySearch/.json?lat={lat}&lon={lon}&radius={radius}&categorySet={categories}&key={tomtom_key}&language=ko-KR"
    
    for attempt in range(retries):
        response = requests.get(url)
        
        if response.status_code == 200:
            try:
                data = response.json()
                results = data.get('results', [])
                if not results:
                    print("No results found. Please check the category IDs and API response.")
                else:
                    # Log the names of the places found
                    for place in results:
                        print(f"Found place: {place['poi']['name']} (Category: {place['poi']['categories']})")
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
            print(f"Response content: {response.text}")
            return []

    print("Error: Exceeded maximum retries for TomTom API")
    return []

def get_user_input_and_recommend():
    while True:
        user_input = input("출발지, 목적지, 테마, 방문할 장소의 수를 입력하세요 (예: 서울역, 강남역, bts, 6): ")
        start, destination, theme, num_visit = user_input.split(", ")

        if start not in df_nodes_essential['name'].values:
          print("해당 출발지가 없습니다. 다시 입력해주세요.")
          continue

        if destination not in df_nodes_essential['name'].values:
          print("해당 목적지가 없습니다. 다시 입력해주세요.")
          continue

        if theme.lower() not in ['bts', '']:
          print("해당 테마가 없습니다. 다시 입력해주세요.")
          continue

        num_visit = int(num_visit)
        if num_visit < 2 or num_visit > 10:
          print("방문할 장소의 수는 2~10이어야 합니다.")
          continue

        start_point = df_nodes_essential[df_nodes_essential['name'] == start][['y', 'x']].values[0]
        destination_point = df_nodes_essential[df_nodes_essential['name'] == destination][['y', 'x']].values[0]

        routes = []

        if theme.lower() == "bts":
            # 출발지와 목적지 사이의 거리 계산
            distance_limit = haversine(start_point, destination_point)
            # 거리 내에 있는 장소만 선택
            bts_visited_places['distance_from_start'] = bts_visited_places.apply(lambda row: haversine(start_point, [row['lat'], row['lon']]), axis=1)
            bts_visited_places['distance_from_destination'] = bts_visited_places.apply(lambda row: haversine(destination_point, [row['lat'], row['lon']]), axis=1)
            filtered_places = bts_visited_places[(bts_visited_places['distance_from_start'] <= distance_limit) & (bts_visited_places['distance_from_destination'] <= distance_limit)]

            # 테마 장소 수 검사 및 조정
            num_visit = min(len(filtered_places), num_visit)
            if len(filtered_places) < num_visit:
                print("테마 장소가 부족합니다. 표시할 수 있는 최대 테마 장소는 {}개입니다.".format(len(filtered_places)))

            # 출발지와의 거리를 기준으로 정렬
            ordered_places = filtered_places.sort_values('distance_from_start')
            visit_points = ordered_places.head(num_visit)

            # Create routes and search for nearby places
            tomtom_key = "GqGLjbuMGbhvTGRIOUf0G4ZStV9AU97j"

            # 출발지 주변의 장소 찾기
            nearby_places_start = search_nearby_places(start_point[0], start_point[1], tomtom_key)
            print(f"출발지 {start} 주변의 장소:")
            for place in nearby_places_start:
                print(f"- {place['poi']['name']} (위도: {place['position']['lat']}, 경도: {place['position']['lon']})")

            for i in range(num_visit):
                routes.append([start_point, visit_points.iloc[i][['lat', 'lon']].values.tolist(), destination_point])
                lat, lon = visit_points.iloc[i][['lat', 'lon']].values.tolist()
                nearby_places = search_nearby_places(lat, lon, tomtom_key)
                print(f"경유지 {i+1} 주변의 장소:")
                for place in nearby_places:
                    print(f"- {place['poi']['name']} (위도: {place['position']['lat']}, 경도: {place['position']['lon']})")

            # 목적지 주변의 장소 찾기
            nearby_places_destination = search_nearby_places(destination_point[0], destination_point[1], tomtom_key)
            print(f"목적지 {destination} 주변의 장소:")
            for place in nearby_places_destination:
                print(f"- {place['poi']['name']} (위도: {place['position']['lat']}, 경도: {place['position']['lon']})")

        else:
            print("지원되지 않는 테마입니다.")
            continue

        print(f"{num_visit}개의 방문 추천 장소는 다음과 같습니다.")
        for i, row in visit_points.iterrows():
            print(f"{row['places_ko']} (위도: {row['lat']}, 경도: {row['lon']})")

        route_segments = []
        route_segments.append(start)
        for i in range(num_visit):
            route_segments.append(visit_points.iloc[i]['places_ko'])
        route_segments.append(destination)
        complete_route = " -> ".join(route_segments)
        print(f"전체 경로: {complete_route}")

        break

    return start_point, destination_point, visit_points[['lat', 'lon']].values

def generate_route_with_chatbot(start_point, destination_point, waypoints):
    # 출발지, 경유지, 목적지를 OpenRouteService가 사용하는 형식으로 변환
    start_point_ors = [start_point[1], start_point[0]]
    destination_point_ors = [destination_point[1], destination_point[0]]
    waypoints_ors = [[point[1], point[0]] for point in waypoints]

    # TomTom API를 사용하여 실시간 교통 정보 가져오기
    tomtom_key = "GqGLjbuMGbhvTGRIOUf0G4ZStV9AU97j"
    traffic_flow_url = f"https://kr-api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={start_point[0]},{start_point[1]}&key={tomtom_key}"
    response = requests.get(traffic_flow_url)
    traffic_data = json.loads(response.text)

    route_url = f"https://kr-api.tomtom.com/routing/1/calculateRoute/{start_point[0]},{start_point[1]}:{destination_point[0]},{destination_point[1]}/json?computeBestOrder=true&key={tomtom_key}"
    response = requests.get(route_url)
    route_data = json.loads(response.text)

    # 경로의 좌표 추출
    points = route_data['routes'][0]['legs'][0]['points']
    coordinates = [(point['latitude'], point['longitude']) for point in points]

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

        # 경유지 주변의 장소를 검색하고 지도에 표시
        nearby_places = search_nearby_places(waypoint[0], waypoint[1], tomtom_key)
        for place in nearby_places:
            folium.Marker(
                location=[place['position']['lat'], place['position']['lon']],
                popup=f"{place['poi']['name']} (Category: {place['poi']['categories']})",
                icon=folium.Icon(color='green', icon='info-sign'),
            ).add_to(m)

    # 출발지 주변의 장소 찾기
    nearby_places_start = search_nearby_places(start_point[0], start_point[1], tomtom_key)
    for place in nearby_places_start:
        folium.Marker(
            location=[place['position']['lat'], place['position']['lon']],
            popup=f"{place['poi']['name']} (Category: {place['poi']['categories']})",
            icon=folium.Icon(color='brown', icon='info-sign'),
        ).add_to(m)

    # 목적지 주변의 장소 찾기
    nearby_places_destination = search_nearby_places(destination_point[0], destination_point[1], tomtom_key)
    for place in nearby_places_destination:
        folium.Marker(
            location=[place['position']['lat'], place['position']['lon']],
            popup=f"{place['poi']['name']} (Category: {place['poi']['categories']})",
            icon=folium.Icon(color='purple', icon='info-sign'),
        ).add_to(m)

    folium.Marker(location=[destination_point[0], destination_point[1]], popup='목적지', icon=folium.Icon(color='red')).add_to(m)

    # 대체 경로 갯수 설정
    alternative_routes_num = 4
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
    print(f"전체 거리: {total_distance/5} km")
    print(f"예상 소요 시간: {int(hours)}시간 {int(minutes)}분")

    return m

# Main loop
while True:
    user_input = get_user_input_and_recommend()
    if user_input is None:
        continue

    start_point, destination_point, num_visit = user_input

    route_map = generate_route_with_chatbot(start_point, destination_point, num_visit)

    route_map.save('map.html')

    choice = input("계속해서 경로를 생성하시겠습니까? (y/n): ")
    if choice.lower() != 'y':
        break