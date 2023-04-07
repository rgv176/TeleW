import requests
import json
import numpy as np
from sklearn.linear_model import LinearRegression

# Point de départ et d'arrivée
start_point = "52.41072,4.84239"
end_point = "52.37919,4.89943"



# Calcul de la distance entre les deux points en utilisant la formule de Haversine
#R = 6373.0  # Rayon de la terre en km
#lat1_rad = radians(lat1)
#lon1_rad = radians(lon1)
#lat2_rad = radians(lat2)
#lon2_rad = radians(lon2)
#dlon = lon2_rad - lon1_rad
#dlat = lat2_rad - lat1_rad
#a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
#c = 2 * atan2(sqrt(a), sqrt(1 - a))
#distance = R * c


# Jours de la semaine
days = ['Thursday', 'Tuesday', 'Wednesday', 'Monday', 'Friday', 'Saturday', 'Sunday']

# Initialiser les listes des temps de trajet et des features
travel_times = []
features = []

# Faire une requête pour chaque jour de la semaine et extraire le temps de trajet et les features
for day in days:
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json?point={start_point}&unit=KMPH&openLr=false&key=OGVoib75TnSfaqaNypxHjxUswEIwwVNk&day={day}"
    response = requests.get(url)
    data = json.loads(response.content)
    traffic_data = data.get('flowSegmentData', None)
    

    if traffic_data is not None:
        traffic_time = traffic_data['currentTravelTime']
        travel_times.append(traffic_time)
        features.append([traffic_data['freeFlowSpeed'], traffic_data['currentSpeed'], traffic_data['confidence']])

# Convertir les listes en arrays numpy
travel_times = np.array(travel_times)
features = np.array(features)

# Créer un modèle de régression linéaire
model = LinearRegression()

# Entrainer le modèle sur les features et les temps de trajet
model.fit(features, travel_times)

# Générer des features pour chaque jour de la semaine et faire des prédictions de temps de trajet
for day in days:
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json?point={start_point}&unit=KMPH&openLr=false&key=OGVoib75TnSfaqaNypxHjxUswEIwwVNk&day={day}"
    response = requests.get(url)
    data = json.loads(response.content)
    traffic_data = data.get('flowSegmentData', None)
    if traffic_data is not None:
        feature = np.array([[traffic_data['freeFlowSpeed'], traffic_data['currentSpeed'], traffic_data['confidence']]])
        travel_time_pred = model.predict(feature)
        travel_times = np.concatenate([travel_times, travel_time_pred])


#Trouver l'index du temps de trajet maximum
max_index = travel_times.argmax()

#Afficher le jour de la semaine correspondant au temps de trajet maximum
print("\n")
print("Le jour de la semaine où il y a le plus d'embouteillages est:", days[max_index])
print("\n")
