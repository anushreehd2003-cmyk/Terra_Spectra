import React from 'react';
import { Layers, MapPin, Activity } from 'lucide-react';

export default function MapView() {
  return (
    <div className="relative w-full h-full bg-slate-900 flex items-center justify-center overflow-hidden">
      {/* Background Geospatial Grid Pattern */}
      <div 
        className="absolute inset-0 opacity-20"
        style={{
          backgroundImage: 'radial-gradient(#10b981 1px, transparent 1px), radial-gradient(#38bdf8 1px, transparent 1px)',
          backgroundSize: '40px 40px',
          backgroundPosition: '0 0, 20px 20px'
        }}
      />

      {/* Center Scaffold Card */}
      <div className="relative z-10 max-w-md p-6 bg-slate-800/80 backdrop-blur-md rounded-xl border border-slate-700 shadow-2xl text-center">
        <div className="w-12 h-12 mx-auto mb-4 bg-emerald-500/20 text-emerald-400 rounded-lg flex items-center justify-center border border-emerald-500/30">
          <Layers className="w-6 h-6 animate-pulse" />
        </div>
        <h2 className="text-xl font-bold text-white mb-2">GIS Canvas Initialized</h2>
        <p className="text-slate-400 text-sm mb-4">
          Deck.gl & Mapbox GL scaffolding active. Ready for farm boundary vector layers and 200-band hyperspectral heatmaps.
        </p>
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-slate-900/60 rounded-full text-xs text-emerald-400 border border-slate-700">
          <MapPin className="w-3.5 h-3.5" />
          <span>Coordinates: 36.3650° N, 120.4300° W (1,000 Acres)</span>
        </div>
      </div>
    </div>
  );
}
