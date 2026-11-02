import React, { useEffect, useRef, useState } from 'react';
import { Deck } from '@deck.gl/core';
import { ScatterplotLayer, LineLayer } from '@deck.gl/layers';
import type { GeoNode } from './GeospatialCommandCenter';

interface Props {
  nodes: GeoNode[];
  selectedRegionId: string;
  onSelectRegion: (id: string) => void;
  flowPhase: number;
  width: number;
  height: number;
  zoom: number;
  onWebGLUnavailable?: () => void;
}

export const DeckGLGeospatialOverlay: React.FC<Props> = ({
  nodes,
  selectedRegionId,
  onSelectRegion,
  flowPhase,
  width,
  height,
  zoom,
  onWebGLUnavailable,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const deckRef = useRef<Deck | null>(null);
  const [webGLSupported, setWebGLSupported] = useState<boolean>(true);

  // Check WebGL availability
  useEffect(() => {
    try {
      const testCanvas = document.createElement('canvas');
      const gl = testCanvas.getContext('webgl2') || testCanvas.getContext('webgl');
      if (!gl) {
        setWebGLSupported(false);
        onWebGLUnavailable?.();
      }
    } catch {
      setWebGLSupported(false);
      onWebGLUnavailable?.();
    }
  }, [onWebGLUnavailable]);

  // Edges data
  const edgesData: Array<{
    source: string;
    target: string;
    sourcePosition: [number, number];
    targetPosition: [number, number];
    flowColor: [number, number, number, number];
  }> = [];

  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const s = nodes[i];
      const t = nodes[j];
      edgesData.push({
        source: s.id,
        target: t.id,
        sourcePosition: [s.lon, s.lat],
        targetPosition: [t.lon, t.lat],
        flowColor: [86, 219, 196, 200],
      });
    }
  }

  // Initialize or update Deck instance
  useEffect(() => {
    if (!canvasRef.current || !webGLSupported) return;

    // ViewState centered on Central Europe
    const viewState = {
      longitude: 7.5,
      latitude: 49.0,
      zoom: 3.5 * zoom,
      pitch: 30,
      bearing: 0,
    };

    const uncertaintyRings = new ScatterplotLayer({
      id: 'uncertainty-rings',
      data: nodes,
      getPosition: (d: GeoNode) => [d.lon, d.lat],
      getRadius: (d: GeoNode) => Math.max(d.uncertaintyWidthMW * 1200, 35000),
      getFillColor: (d: GeoNode) =>
        d.status === 'OUTAGE' ? [240, 113, 103, 70] : [86, 219, 196, 50],
      getLineColor: (d: GeoNode) =>
        d.status === 'OUTAGE' ? [240, 113, 103, 180] : [86, 219, 196, 160],
      stroked: true,
      filled: true,
      lineWidthMinPixels: 1.5,
      pickable: false,
    });

    const transmissionLines = new LineLayer({
      id: 'transmission-corridors',
      data: edgesData,
      getSourcePosition: (d) => d.sourcePosition,
      getTargetPosition: (d) => d.targetPosition,
      getColor: (d) => d.flowColor,
      getWidth: 3,
      widthMinPixels: 2,
      pickable: false,
    });

    const nodePoints = new ScatterplotLayer({
      id: 'sensor-nodes',
      data: nodes,
      getPosition: (d: GeoNode) => [d.lon, d.lat],
      getRadius: (d: GeoNode) => (d.id === selectedRegionId ? 28000 : 20000),
      getFillColor: (d: GeoNode) => {
        if (d.status === 'OUTAGE') return [200, 40, 40, 240];
        if (d.id === selectedRegionId) return [22, 141, 128, 255];
        return [20, 60, 74, 220];
      },
      getLineColor: (d: GeoNode) =>
        d.id === selectedRegionId ? [86, 219, 196, 255] : [115, 165, 175, 200],
      stroked: true,
      filled: true,
      lineWidthMinPixels: 2,
      pickable: true,
      onClick: (info) => {
        if (info.object) {
          onSelectRegion((info.object as GeoNode).id);
        }
      },
    });

    if (!deckRef.current) {
      try {
        deckRef.current = new Deck({
          canvas: canvasRef.current,
          initialViewState: viewState,
          controller: true,
          layers: [transmissionLines, uncertaintyRings, nodePoints],
        });
      } catch {
        setWebGLSupported(false);
        onWebGLUnavailable?.();
      }
    } else {
      deckRef.current.setProps({
        initialViewState: viewState,
        layers: [transmissionLines, uncertaintyRings, nodePoints],
      });
    }
  }, [nodes, selectedRegionId, flowPhase, zoom, webGLSupported, onSelectRegion, onWebGLUnavailable]);

  // Clean up Deck on unmount
  useEffect(() => {
    return () => {
      if (deckRef.current) {
        deckRef.current.finalize();
        deckRef.current = null;
      }
    };
  }, []);

  if (!webGLSupported) {
    return null; // Signals fallback to SVG/Canvas
  }

  return (
    <div
      className="deckgl-container"
      style={{ position: 'relative', width: '100%', height: `${height}px` }}
      role="region"
      aria-label="Deck.gl WebGL Geospatial Transmission Overlay"
    >
      <canvas
        ref={canvasRef}
        id="deckgl-overlay-canvas"
        style={{ width: '100%', height: '100%', display: 'block', borderRadius: '8px' }}
      />
      <div className="deckgl-attribution-tag" aria-hidden="true">
        RENDERER: DECK.GL / WEBGL (ACCELERATED)
      </div>
    </div>
  );
};
