"""
Layer 4: Spatial Consistency Model.
Performs Inverse Distance Weighting (IDW) interpolation with elevation lapse-rate correction
across k-nearest AWS neighbors to detect isolated station anomalies vs regional weather fronts.
"""
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.ingestion.base import WeatherReading, StationInfo


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two GPS coordinates in kilometers."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * (math.sin(dlon / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


class SpatialConsistencyModel:
    """
    Computes spatial consensus among neighboring Automatic Weather Stations.
    Adjusts temperature for elevation differences using standard environmental lapse rate (6.5°C / km).
    """

    def __init__(self, k_neighbors: int = 4, max_distance_km: float = 650.0):
        self.k_neighbors = k_neighbors
        self.max_distance_km = max_distance_km
        self.lapse_rate_c_per_km = 6.5

    def find_nearest_neighbors(
        self,
        target_station: StationInfo,
        all_stations: List[StationInfo]
    ) -> List[Tuple[StationInfo, float]]:
        """Finds k-nearest stations sorted by haversine distance."""
        distances = []
        for s in all_stations:
            if s.id == target_station.id:
                continue
            dist = haversine_distance_km(
                target_station.latitude, target_station.longitude,
                s.latitude, s.longitude
            )
            distances.append((s, dist))

        distances.sort(key=lambda x: x[1])
        return distances[:self.k_neighbors]

    def evaluate(
        self,
        reading: WeatherReading,
        target_station: StationInfo,
        all_stations: List[StationInfo],
        current_readings_map: Dict[str, WeatherReading],
        is_cross_station_check: bool = False
    ) -> Dict[str, Any]:
        """
        Calculates IDW elevation-corrected spatial expectation and residual for target station.
        """
        neighbors = self.find_nearest_neighbors(target_station, all_stations)
        if not neighbors or reading.temperature is None:
            return {
                "score": 0.0,
                "spatial_expected_temp": reading.temperature,
                "neighbor_agreement": 1.0,
                "neighbors_count": 0,
                "neighbor_temps": []
            }

        weights = []
        corrected_temps = []
        neighbor_data = []

        for stn, dist in neighbors:
            nbr_reading = current_readings_map.get(stn.id)
            if nbr_reading and nbr_reading.temperature is not None:
                # Lapse-rate elevation adjustment:
                # Elev difference = target.elevation - neighbor.elevation
                # If target is higher, target should be cooler: delta_T = -6.5 * (elev_diff / 1000)
                elev_diff_km = (target_station.elevation_m - stn.elevation_m) / 1000.0
                t_adj = nbr_reading.temperature - (self.lapse_rate_c_per_km * elev_diff_km)

                w = 1.0 / max(dist, 10.0) ** 1.5
                weights.append(w)
                corrected_temps.append(t_adj)
                neighbor_data.append({
                    "station_id": stn.id,
                    "name": stn.name,
                    "distance_km": round(dist, 1),
                    "raw_temp": nbr_reading.temperature,
                    "elev_corrected_temp": round(t_adj, 1)
                })

        if not corrected_temps:
            return {
                "score": 0.0,
                "spatial_expected_temp": reading.temperature,
                "neighbor_agreement": 1.0,
                "neighbors_count": 0,
                "neighbor_temps": []
            }

        weights = np.array(weights)
        weights /= np.sum(weights)

        spatial_expected_t = float(np.sum(weights * np.array(corrected_temps)))
        temp_residual = abs(reading.temperature - spatial_expected_t)

        # Spatial residual score: 0 if residual <= 2.5°C, rising to 1.0 if residual >= 12°C
        spatial_score = float(np.clip((temp_residual - 2.5) / 9.5, 0.0, 1.0))

        # Check neighbor agreement: how many neighbors confirm a similar reading?
        # If target has 55°C and all neighbors report 30-33°C, agreement is 0.0 (high spatial outlier)
        diffs = [abs(reading.temperature - ct) for ct in corrected_temps]
        supporting_neighbors = sum(1 for d in diffs if d <= 4.0)
        neighbor_agreement = supporting_neighbors / len(corrected_temps)

        # Cross-Station Verification:
        # Compare nearby AWS stations.
        # If only one station shows an extreme value, classify it as a possible sensor fault;
        # if multiple nearby stations show the same abnormal pattern, flag it as a possible real weather event.
        is_spatial_outlier = spatial_score > 0.65 and neighbor_agreement < 0.25
        is_extreme_target = temp_residual >= 4.0 or reading.temperature >= 48.0 or reading.temperature <= 0.0

        abnormal_agreeing_neighbors = []
        nominal_neighbors = []
        for n_item, ct in zip(neighbor_data, corrected_temps):
            d = abs(reading.temperature - ct)
            if d <= 4.0:
                abnormal_agreeing_neighbors.append(n_item["name"])
            else:
                nominal_neighbors.append(f"{n_item['name']} ({ct:.1f}°C)")

        if is_cross_station_check or is_extreme_target or spatial_score >= 0.5:
            # Check if multiple nearby stations show the same abnormal pattern / extreme reading
            if len(abnormal_agreeing_neighbors) >= 2 or (len(corrected_temps) <= 2 and len(abnormal_agreeing_neighbors) >= 1) or neighbor_agreement >= 0.5:
                cross_station_verdict = "POSSIBLE_REAL_WEATHER_EVENT"
                classification = "Possible Real Weather Event"
                is_sensor_fault = False
                is_real_weather_event = True
                verdict_detail = (
                    f"Multiple nearby stations ({', '.join(abnormal_agreeing_neighbors)}) show the same abnormal pattern "
                    f"({reading.temperature:.1f}°C). High spatial consensus ({neighbor_agreement*100:.0f}%) indicates "
                    f"a possible real weather event (e.g. convective front, thunderstorm, or regional weather pattern)."
                )
            else:
                cross_station_verdict = "POSSIBLE_SENSOR_FAULT"
                classification = "Possible Sensor Fault"
                is_sensor_fault = True
                is_real_weather_event = False
                verdict_detail = (
                    f"Only target station {target_station.id} ({target_station.name}) shows an extreme value of {reading.temperature:.1f}°C "
                    f"(residual: +{temp_residual:.1f}°C vs expected {spatial_expected_t:.1f}°C). Nearby stations ({', '.join(nominal_neighbors[:3])}) "
                    f"remain nominal. Classifying as a possible sensor fault."
                )
        else:
            cross_station_verdict = "NOMINAL"
            classification = "Consistent / Normal"
            is_sensor_fault = False
            is_real_weather_event = False
            verdict_detail = f"Station reading aligns with {len(corrected_temps)} nearby stations within normal atmospheric limits."

        cross_station_data = {
            "verdict": cross_station_verdict,
            "classification": classification,
            "is_sensor_fault": is_sensor_fault,
            "is_real_weather_event": is_real_weather_event,
            "target_station_id": target_station.id,
            "target_station_name": target_station.name,
            "target_value": reading.temperature,
            "expected_value": round(spatial_expected_t, 1),
            "temp_residual": round(temp_residual, 2),
            "nearby_stations_count": len(corrected_temps),
            "agreeing_stations_count": len(abnormal_agreeing_neighbors) if (is_cross_station_check or is_extreme_target or spatial_score >= 0.5) else supporting_neighbors,
            "deviating_stations_count": len(corrected_temps) - supporting_neighbors,
            "nearby_stations": neighbor_data,
            "verdict_detail": verdict_detail
        }

        return {
            "score": round(spatial_score, 3),
            "spatial_expected_temp": round(spatial_expected_t, 1),
            "temp_residual": round(temp_residual, 2),
            "neighbor_agreement": round(neighbor_agreement, 2),
            "neighbors_count": len(corrected_temps),
            "neighbor_data": neighbor_data,
            "is_spatial_outlier": is_spatial_outlier,
            "cross_station_verification": cross_station_data
        }

    def verify_cross_station(
        self,
        reading: WeatherReading,
        target_station: StationInfo,
        all_stations: List[StationInfo],
        current_readings_map: Dict[str, WeatherReading]
    ) -> Dict[str, Any]:
        """Direct helper to perform cross-station verification comparing nearby AWS stations."""
        result = self.evaluate(reading, target_station, all_stations, current_readings_map, is_cross_station_check=True)
        return result.get("cross_station_verification", {})
