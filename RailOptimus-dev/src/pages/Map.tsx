
import React, { useState, useEffect, useMemo } from 'react';
import { MapContainer, TileLayer, GeoJSON ,useMap} from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import axios from 'axios';
import { useAuthStore } from '../stores/authStore';
import L from 'leaflet'; // Import the main Leaflet library to create custom icons
import StationIconUrl from '../assets/pin.png';
// --- 1. STYLING FUNCTION FOR TRACKS (LineString) ---
// This function is only called for features with a path, like lines.
const getTrackStyle = (feature) => {
  const props = feature?.properties || {};
  const service = (props.service || "").toLowerCase();
  const usage = (props.usage || "").toLowerCase();
  
  let color = "#808080"; // Default grey color

  if (service === "yard") color = "#0000FF";
  else if (service === "siding") color = "#008000";
  else if (service === "spur") color = "#800080";
  else if(service ==="crossover") color ="#FFFF00"
  else if (usage === "branch") color = "#FFA500";
  else if (usage === "main") color = "#FF0000";

  return { color: color, weight: 3, opacity: 0.85 };
};

// --- 2. MARKER FUNCTION FOR STATIONS (Point) ---
// This function is only called for Point features.
const createStationMarker = (feature, latlng) => {
  const stationIcon = new L.Icon({
    iconUrl: StationIconUrl, // A generic train station icon
    iconSize: [25, 25],
    iconAnchor: [12, 25],
    popupAnchor: [0, -25]
  });

  const marker = L.marker(latlng, { icon: stationIcon });

  // Add a popup with the station's name from its properties
  const stationName = feature.properties?.name || 'Unknown Station';
  marker.bindPopup(`<b>Station:</b> ${stationName}`);

  return marker;
};

// --- 3. DATA FOR THE LEGEND ---
const legendItems = [
    { color: "#FF0000", label: "Main Line" },
    { color: "#FFA500", label: "Branch Line" },
    { color: "#0000FF", label: "Yard" },
    { color: "#008000", label: "Siding" },
    { color: "#800080", label: "Spur" },
    { color: "#FFFF00",  label:"Crossover" }
  ];

const AutoFitBounds: React.FC<{ data: any }> = ({ data }) => {
  const map = useMap(); // This now works because of the import fix

  useEffect(() => {
    if (!data || !data.features || data.features.length === 0) {
      return;
    }
    const geoJsonLayer = L.geoJSON(data);
    const bounds = geoJsonLayer.getBounds();
    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [data, map]);

  return null;
};

const FeatureCounter: React.FC<{ data: any }> = ({ data }) => {
  const counts = useMemo(() => {
    if (!data || !data.features) {
      return { tracks: 0, stations: 0 };
    }
    let tracks = 0;
    let stations = 0;
    for (const feature of data.features) {
      const geometryType = feature?.geometry?.type;
      if (geometryType === 'LineString') {
        tracks++;
      } else if (geometryType === 'Point') {
        stations++;
      }
    }
    return { tracks, stations };
  }, [data]);
  
  // The 'return' keyword is what fixes the 'void' error.
  return (
    <div style={{ padding: '10px 0', borderBottom: '1px solid #ddd', marginBottom: '1rem' }}>
      <p><b>Total Tracks:</b> {counts.tracks}</p>
      <p><b>Total Stations:</b> {counts.stations}</p>
    </div>
  );
};


const MapPage: React.FC = () => {
  const user = useAuthStore((state) => state.user);
  const [mapData, setMapData] = useState<any>(null);
  const [trackCount, setTrackCount] = useState<number>(0);
  const [stationCount, setStationCount] = useState<number>(0);
  const [selectedFeature, setSelectedFeature] = useState<any>(null);


  useEffect(() => {
    if (user?.section) {
      const fetchMapData = async () => {
        try {
          const response = await axios.get(`http://localhost:5000/api/maps/mapdata/${user.section.toLowerCase()}`);
          const data = response.data;   // ✅ extract response data
          setMapData(data);
        } catch (error) {
          console.error("Could not fetch map data", error);
          setMapData(null);
        }
      };
      fetchMapData();
    }
  }, [user]);
 // --- 2. NEW: Function to attach event listeners to each map feature ---
  const onEachFeature = (feature, layer) => {
    layer.on({
      click: () => {
        // When a feature is clicked, update the state with its properties
        setSelectedFeature(feature.properties);
      }
    });
  };


  if (!user || !mapData) {
    return <div className="flex items-center justify-center h-full text-xl">Loading Map...</div>;
  }

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100%' }}>
      <MapContainer center={[27.17, 78.00]} zoom={10} style={{ height: '100%', flex: 1 }}>
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'/>
        
        {/* 4. THE GEOJSON COMPONENT NOW USES BOTH PROPS */}
        <GeoJSON 
          key={user.section} // Add a key to force re-render when the user/data changes
          data={mapData} 
          style={getTrackStyle} // This will apply to LineStrings
          pointToLayer={createStationMarker} // This will apply to Points
          onEachFeature={onEachFeature} // event listener for interactivity
        />
        <AutoFitBounds data={mapData} />
      </MapContainer>
      
      {/* --- 5. SIDEBAR WITH LEGEND --- */}
      {/* --- 4. UPDATED SIDEBAR: Conditionally renders Legend or Feature Info --- */}
      <div style={{ width: '250px', padding: '1rem', overflowY: 'auto', borderLeft: '1px solid #ddd', backgroundColor: '#f9f9f' }}>
        {selectedFeature ? (
          // --- If a feature is selected, show its properties ---
          <div>
            <button onClick={() => setSelectedFeature(null)} style={{ marginBottom: '1rem', cursor: 'pointer' }}>
              &larr; Back to Legend
            </button>
            <div style={{ backgroundColor: '#eef2f5', borderRadius: '8px', padding: '12px', border: '1px solid #dee2e6' }}>
              <h3 style={{ fontSize: "1rem", marginBottom: "1rem", marginTop: 0 }}>Feature Details</h3>
              <ul style={{ listStyle: "none", padding: 0 }}>
                {Object.entries(selectedFeature).map(([key, value]) => (
                  <li key={key} style={{ marginBottom: '5px', wordBreak: 'break-word' }}>
                    <strong style={{ textTransform: 'capitalize' }}>{key.replace(/_/g, ' ')}:</strong> {String(value)}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        ):(
      <div style={{ width: '250px', padding: '1rem', overflowY: 'auto', borderLeft: '1px solid #ddd', backgroundColor: '#f9f9f9' }}>
        <h2 style={{ fontSize: '1.2rem', marginBottom: '0.5rem' }}>{user.section.toUpperCase()} Section</h2>
        <p style={{ marginBottom: '1.5rem', borderBottom: '1px solid #ddd', paddingBottom: '1rem' }}>Controller: {user.name}</p>
        {/* --- Component to display counts is used here --- */}
        <FeatureCounter data= {mapData} />  
        <h3 style={{ fontSize: "1rem", marginBottom: "1rem" }}>Legend</h3>
        <ul style={{ listStyle: "none", padding: 0 }}>
          {legendItems.map(item => (
            <li key={item.label} style={{ marginBottom: '8px', display: 'flex', alignItems: 'center' }}>
              <span style={{ backgroundColor: item.color, width: '25px', height: '5px', marginRight: '10px', display: 'inline-block', border: '1px solid #555' }}></span>
              {item.label}
            </li>
          ))}
          {/* Legend for stations */}
          <li style={{ marginTop: '12px', display: 'flex', alignItems: 'center' }}>
            <img src={StationIconUrl} alt="station icon" style={{width: '20px', height: '20px', marginRight: '10px'}}/>
            Station
          </li>
        </ul>
        </div>
        )}
      </div>
    </div>
  );
};

export default MapPage;

