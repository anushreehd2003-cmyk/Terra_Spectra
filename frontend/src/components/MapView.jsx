import React, { useState, useEffect } from 'react';
import { Layers, MapPin, AlertCircle, CheckCircle2, ShieldAlert, Eye, RefreshCw } from 'lucide-react';
import apiService from '../services/api';

export default function MapView({ onSelectParcel }) {
  const [geoData, setGeoData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedParcel, setSelectedParcel] = useState(null);
  const [showHeatmap, setShowHeatmap] = useState(true);

  useEffect(() => {
    async function loadGeoJSON() {
      try {
        const data = await apiService.getFarmGeoJSON();
        setGeoData(data);
        if (data?.features?.length > 0) {
          setSelectedParcel(data.features[1]); // Default to Zone B (warning)
        }
      } catch (err) {
        console.error('Failed to load farm GeoJSON:', err);
      } finally {
        setLoading(false);
      }
    }
    loadGeoJSON();
  }, []);

  const handleSelect = (feature) => {
    setSelectedParcel(feature);
    if (onSelectParcel) onSelectParcel(feature);
  };

  return (
    <div className="relative w-full h-full bg-slate-950 flex flex-col overflow-hidden">
      {/* Background Geospatial Grid Pattern */}
      <div 
        className="absolute inset-0 opacity-25"
        style={{
          backgroundImage: 'radial-gradient(#10b981 1.5px, transparent 1.5px), radial-gradient(#38bdf8 1px, transparent 1px)',
          backgroundSize: '40px 40px',
          backgroundPosition: '0 0, 20px 20px'
        }}
      />

      {/* Top Map HUD Overlay */}
      <div className="relative z-10 p-4 flex items-center justify-between pointer-events-none">
        <div className="pointer-events-auto bg-slate-900/90 backdrop-blur border border-slate-800 rounded-lg px-3 py-2 flex items-center gap-3 shadow-lg">
          <div className="p-1.5 bg-emerald-500/20 text-emerald-400 rounded-md">
            <Layers className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-white">TerraSpectra Valley Farm GIS</h3>
            <p className="text-[10px] text-slate-400">1,000-Acre Plot • 36.3650° N, 120.4300° W • WGS84</p>
          </div>
        </div>

        <div className="pointer-events-auto flex items-center gap-2">
          <button
            onClick={() => setShowHeatmap(!showHeatmap)}
            className={`px-3 py-1.5 text-xs rounded-lg border flex items-center gap-1.5 transition ${
              showHeatmap
                ? 'bg-amber-500/20 border-amber-500/40 text-amber-300'
                : 'bg-slate-900/90 border-slate-800 text-slate-400'
            }`}
          >
            <Eye className="w-3.5 h-3.5" />
            {showHeatmap ? 'Risk Gradient Visible' : 'Risk Gradient Hidden'}
          </button>
        </div>
      </div>

      {/* Interactive Agricultural Parcels Canvas */}
      <div className="relative z-10 flex-1 flex items-center justify-center p-6">
        <div className="w-full max-w-2xl aspect-video bg-slate-900/80 backdrop-blur-md rounded-2xl border border-slate-800 p-6 flex flex-col justify-between shadow-2xl relative overflow-hidden">
          {/* Subtle Radar Scan Line Animation */}
          <div className="absolute inset-0 bg-gradient-to-b from-transparent via-emerald-500/5 to-transparent h-24 w-full animate-pulse pointer-events-none" />

          {/* 4-Zone Farm Layout Grid */}
          <div className="grid grid-cols-2 grid-rows-2 gap-4 h-full w-full">
            {geoData?.features?.map((feature) => {
              const isSelected = selectedParcel?.id === feature.id;
              const isHighRisk = feature.properties.risk_score > 0.5;

              return (
                <div
                  key={feature.id}
                  onClick={() => handleSelect(feature)}
                  className={`relative rounded-xl p-4 cursor-pointer transition-all duration-200 flex flex-col justify-between border ${
                    isSelected
                      ? 'ring-2 ring-emerald-400 border-transparent shadow-lg shadow-emerald-500/10'
                      : 'hover:border-slate-600'
                  } ${
                    isHighRisk
                      ? showHeatmap
                        ? 'bg-gradient-to-br from-amber-950/40 via-slate-900 to-amber-900/20 border-amber-800/50'
                        : 'bg-slate-800/40 border-slate-700/60'
                      : 'bg-emerald-950/20 border-emerald-800/30'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400">
                        {feature.properties.crop}
                      </span>
                      <h4 className="text-sm font-bold text-white mt-0.5">
                        {feature.properties.name}
                      </h4>
                    </div>
                    {isHighRisk ? (
                      <span className="p-1.5 bg-amber-500/20 text-amber-400 rounded-lg">
                        <ShieldAlert className="w-4 h-4" />
                      </span>
                    ) : (
                      <span className="p-1.5 bg-emerald-500/20 text-emerald-400 rounded-lg">
                        <CheckCircle2 className="w-4 h-4" />
                      </span>
                    )}
                  </div>

                  <div className="mt-3 flex items-center justify-between text-xs">
                    <span className="text-slate-400">Pre-Symptomatic Risk:</span>
                    <span
                      className={`font-mono font-bold ${
                        isHighRisk ? 'text-amber-400' : 'text-emerald-400'
                      }`}
                    >
                      {(feature.properties.risk_score * 100).toFixed(0)}%
                    </span>
                  </div>

                  {/* Progress bar */}
                  <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden mt-1.5">
                    <div
                      className={`h-full rounded-full transition-all duration-300 ${
                        isHighRisk
                          ? 'bg-gradient-to-r from-amber-500 to-rose-500'
                          : 'bg-emerald-500'
                      }`}
                      style={{ width: `${feature.properties.risk_score * 100}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          {/* Bottom parcel metadata bar */}
          {selectedParcel && (
            <div className="mt-4 pt-3 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
              <div className="flex items-center gap-2">
                <MapPin className="w-3.5 h-3.5 text-emerald-400" />
                <span className="text-white font-medium">{selectedParcel.properties.name}</span>
                <span>• Status: <strong className="text-slate-200">{selectedParcel.properties.status}</strong></span>
              </div>
              <span className="text-[11px] text-slate-500">
                Click zone to inspect spectral details
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
