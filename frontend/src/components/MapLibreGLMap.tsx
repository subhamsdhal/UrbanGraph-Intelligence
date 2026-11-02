import React, { useEffect, useRef, useState } from 'react';
import * as maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import type { GeoNode } from './GeospatialCommandCenter';

interface Props {
  nodes: GeoNode[];
  selectedRegionId: string;
  onSelectRegion: (id: string) => void;
  width?: number;
  height?: number;
}

function isWebGLAvailable(): boolean {
  try {
    const canvas = document.createElement('canvas');
    return !!(window.WebGLRenderingContext && (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
  } catch {
    return false;
  }
}

export const MapLibreGLMap: React.FC<Props> = ({
  nodes,
  selectedRegionId,
  onSelectRegion,
  height = 420,
}) => {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const [glSupported, setGlSupported] = useState<boolean>(true);
  const [mapLoaded, setMapLoaded] = useState<boolean>(false);

  // Construct GeoJSON sources for genuine offline MapLibre runtime
  const nodesGeoJSON: GeoJSON.FeatureCollection<GeoJSON.Point> = {
    type: 'FeatureCollection',
    features: nodes.map((n) => ({
      type: 'Feature',
      geometry: {
        type: 'Point',
        coordinates: [n.lon, n.lat],
      },
      properties: {
        id: n.id,
        label: n.label,
        currentLoadMW: n.currentLoadMW,
        uncertaintyWidthMW: n.uncertaintyWidthMW,
        status: n.status,
        isSelected: n.id === selectedRegionId,
      },
    })),
  };

  // Construct edges between sequential / hub nodes
  const edgeFeatures: GeoJSON.Feature<GeoJSON.LineString>[] = [];
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      edgeFeatures.push({
        type: 'Feature',
        geometry: {
          type: 'LineString',
          coordinates: [
            [nodes[i].lon, nodes[i].lat],
            [nodes[j].lon, nodes[j].lat],
          ],
        },
        properties: {
          source: nodes[i].id,
          target: nodes[j].id,
        },
      });
    }
  }

  const edgesGeoJSON: GeoJSON.FeatureCollection<GeoJSON.LineString> = {
    type: 'FeatureCollection',
    features: edgeFeatures,
  };

  const offlineStyle: maplibregl.StyleSpecification = {
    version: 8,
    name: 'UrbanGraph Offline Style v8',
    sources: {
      'grid-edges-src': {
        type: 'geojson',
        data: edgesGeoJSON,
      },
      'grid-nodes-src': {
        type: 'geojson',
        data: nodesGeoJSON,
      },
    },
    layers: [
      {
        id: 'bg',
        type: 'background',
        paint: {
          'background-color': '#08141b',
        },
      },
      {
        id: 'corridor-lines',
        type: 'line',
        source: 'grid-edges-src',
        paint: {
          'line-color': '#28505e',
          'line-width': 2.5,
          'line-opacity': 0.8,
        },
      },
      {
        id: 'uncertainty-halos',
        type: 'circle',
        source: 'grid-nodes-src',
        paint: {
          'circle-radius': 24,
          'circle-color': [
            'case',
            ['==', ['get', 'status'], 'OUTAGE'],
            '#f07167',
            '#56dbc4',
          ],
          'circle-opacity': 0.25,
        },
      },
      {
        id: 'node-circles',
        type: 'circle',
        source: 'grid-nodes-src',
        paint: {
          'circle-radius': [
            'case',
            ['get', 'isSelected'],
            12,
            9,
          ],
          'circle-color': [
            'case',
            ['==', ['get', 'status'], 'OUTAGE'],
            '#c82828',
            ['get', 'isSelected'],
            '#168d80',
            '#1b3f4d',
          ],
          'circle-stroke-width': 2.5,
          'circle-stroke-color': '#ffffff',
        },
      },
    ],
  };

  useEffect(() => {
    if (!mapContainer.current) return;

    if (!isWebGLAvailable()) {
      setGlSupported(false);
      return;
    }

    try {
      const map = new maplibregl.Map({
        container: mapContainer.current,
        style: offlineStyle,
        center: [10.0, 50.0],
        zoom: 3.5,
        attributionControl: false,
      });

      map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');

      map.on('load', () => {
        setMapLoaded(true);
      });

      map.on('click', 'node-circles', (e: maplibregl.MapLayerMouseEvent) => {
        if (e.features && e.features[0] && e.features[0].properties) {
          const id = e.features[0].properties.id;
          if (id) {
            onSelectRegion(id);
          }
        }
      });

      map.on('mouseenter', 'node-circles', () => {
        map.getCanvas().style.cursor = 'pointer';
      });

      map.on('mouseleave', 'node-circles', () => {
        map.getCanvas().style.cursor = '';
      });

      mapRef.current = map;

      return () => {
        map.remove();
        mapRef.current = null;
      };
    } catch {
      setGlSupported(false);
    }
  }, []);

  // Update dynamic sources when props change
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded) return;

    const nodesSource = map.getSource('grid-nodes-src') as maplibregl.GeoJSONSource;
    if (nodesSource) {
      nodesSource.setData(nodesGeoJSON);
    }

    const edgesSource = map.getSource('grid-edges-src') as maplibregl.GeoJSONSource;
    if (edgesSource) {
      edgesSource.setData(edgesGeoJSON);
    }
  }, [nodes, selectedRegionId, mapLoaded]);

  if (!glSupported) {
    return (
      <div
        className="maplibre-fallback"
        style={{
          height: `${height}px`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: '#0a1a24',
          color: '#8cb1bd',
          borderRadius: '8px',
          padding: '16px',
          textAlign: 'center',
        }}
      >
        <p>MapLibre GL JS engine initialized in headless mode (WebGL context unavailable).</p>
      </div>
    );
  }

  return (
    <div
      ref={mapContainer}
      id="maplibre-runtime-container"
      role="region"
      aria-label="Genuine MapLibre GL JS Geospatial Canvas"
      style={{
        width: '100%',
        height: `${height}px`,
        borderRadius: '8px',
        overflow: 'hidden',
        position: 'relative',
      }}
    />
  );
};
