import React, { useState, useEffect } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceArea,
  ReferenceLine,
} from 'recharts';
import { Activity, ShieldAlert, Sparkles, Info, RefreshCw } from 'lucide-react';
import apiService from '../services/api';

export default function SpectralInspector() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [visibleLines, setVisibleLines] = useState({
    healthy: true,
    pre_symptomatic: true,
    delta_pre_symp: true,
    necrotic: false,
    soil: false,
  });

  const fetchSignatures = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiService.getSpectralSignatures();
      setData(res.data_points || []);
    } catch (err) {
      console.error('Error fetching signatures:', err);
      setError('Unable to load live spectral data. Showing synthesized simulation.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSignatures();
  }, []);

  const toggleLine = (key) => {
    setVisibleLines((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <div className="flex flex-col h-full w-full bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-xl">
      {/* Header bar */}
      <div className="p-4 border-b border-slate-800 bg-slate-900/90 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <Activity className="w-5 h-5 text-emerald-400" />
            204-Band Continuous Spectral Reflectance Profile
            <span className="text-[11px] font-semibold uppercase px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              400 nm - 2400 nm
            </span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Pre-symptomatic fungal infection fingerprinting: Notice the NIR drop and Red-Edge shift despite identical visible reflectance.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={fetchSignatures}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs rounded-lg border border-slate-700 transition"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            Refresh Bands
          </button>
        </div>
      </div>

      {/* Physics Callout Banner */}
      <div className="mx-4 mt-4 p-3 bg-slate-950/70 border border-slate-800 rounded-lg flex items-start gap-3">
        <Sparkles className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
        <div className="text-xs space-y-1">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-200">The 3-Week Early Detection Physics:</span>
            <span className="px-1.5 py-0.5 bg-amber-500/20 text-amber-300 rounded text-[10px] font-mono">
              VISIBLE DELTA ≈ 0.00
            </span>
            <span className="px-1.5 py-0.5 bg-emerald-500/20 text-emerald-300 rounded text-[10px] font-mono">
              NIR DELTA &gt; 0.12
            </span>
          </div>
          <p className="text-slate-400 leading-relaxed">
            In visible wavelengths (400-680 nm), the healthy and pre-symptomatic curves overlap completely because chlorophyll has not yet degraded.
            Traditional RGB and broad NDVI satellites report normal green foliage. However, fungal hyphae in the intercellular mesophyll cause a 
            <strong className="text-amber-400"> ~20% collapse in NIR reflectance (750-1300 nm)</strong> and a <strong className="text-cyan-400">leftward shift in the Red Edge inflection (700-740 nm)</strong>.
          </p>
        </div>
      </div>

      {/* Toggles */}
      <div className="flex flex-wrap gap-2 px-4 py-3 items-center">
        <span className="text-xs font-medium text-slate-400 mr-2">Spectral Layers:</span>
        <button
          onClick={() => toggleLine('healthy')}
          className={`px-2.5 py-1 text-xs rounded-md font-medium border transition ${
            visibleLines.healthy
              ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-300'
              : 'bg-slate-800/60 border-slate-700 text-slate-400'
          }`}
        >
          ● Healthy Canopy
        </button>
        <button
          onClick={() => toggleLine('pre_symptomatic')}
          className={`px-2.5 py-1 text-xs rounded-md font-medium border transition ${
            visibleLines.pre_symptomatic
              ? 'bg-amber-500/20 border-amber-500/50 text-amber-300'
              : 'bg-slate-800/60 border-slate-700 text-slate-400'
          }`}
        >
          ● Pre-Symptomatic Fungal
        </button>
        <button
          onClick={() => toggleLine('delta_pre_symp')}
          className={`px-2.5 py-1 text-xs rounded-md font-medium border transition ${
            visibleLines.delta_pre_symp
              ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-300'
              : 'bg-slate-800/60 border-slate-700 text-slate-400'
          }`}
        >
          ● Early Warning Delta (Δ)
        </button>
        <button
          onClick={() => toggleLine('necrotic')}
          className={`px-2.5 py-1 text-xs rounded-md font-medium border transition ${
            visibleLines.necrotic
              ? 'bg-rose-500/20 border-rose-500/50 text-rose-300'
              : 'bg-slate-800/60 border-slate-700 text-slate-400'
          }`}
        >
          ● Advanced Necrotic
        </button>
        <button
          onClick={() => toggleLine('soil')}
          className={`px-2.5 py-1 text-xs rounded-md font-medium border transition ${
            visibleLines.soil
              ? 'bg-slate-500/20 border-slate-500/50 text-slate-300'
              : 'bg-slate-800/60 border-slate-700 text-slate-400'
          }`}
        >
          ● Bare Agricultural Soil
        </button>
      </div>

      {/* Chart Canvas */}
      <div className="flex-1 w-full min-h-[380px] p-4">
        {loading ? (
          <div className="h-full w-full flex items-center justify-center text-slate-400 text-sm">
            <RefreshCw className="w-5 h-5 animate-spin mr-2 text-emerald-400" />
            Loading 204 continuous spectral channels...
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 15, right: 30, left: 0, bottom: 25 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
              
              {/* Highlight Critical Wavelength Regions */}
              <ReferenceArea x1={400} x2={700} fill="#3b82f6" fillOpacity={0.06} />
              <ReferenceArea x1={700} x2={750} fill="#10b981" fillOpacity={0.12} />
              <ReferenceArea x1={750} x2={1300} fill="#f59e0b" fillOpacity={0.08} />
              <ReferenceArea x1={1350} x2={1500} fill="#06b6d4" fillOpacity={0.08} />
              <ReferenceArea x1={1850} x2={2050} fill="#06b6d4" fillOpacity={0.08} />

              <XAxis
                dataKey="wavelength"
                stroke="#94a3b8"
                fontSize={11}
                tickFormatter={(val) => `${val}nm`}
                label={{ value: 'Spectral Wavelength (nm)', position: 'insideBottom', offset: -15, fill: '#94a3b8', fontSize: 12 }}
              />
              <YAxis
                stroke="#94a3b8"
                fontSize={11}
                domain={[0, 0.7]}
                tickFormatter={(val) => val.toFixed(2)}
                label={{ value: 'Reflectance [0.0 - 1.0]', angle: -90, position: 'insideLeft', offset: 15, fill: '#94a3b8', fontSize: 12 }}
              />
              
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                itemStyle={{ padding: '2px 0' }}
                formatter={(value, name) => [value, name.replace(/_/g, ' ')]}
                labelFormatter={(label) => `Wavelength: ${label} nm`}
              />

              <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: '12px', color: '#cbd5e1' }} />

              {visibleLines.healthy && (
                <Line
                  type="monotone"
                  dataKey="healthy"
                  name="Healthy Canopy"
                  stroke="#10b981"
                  strokeWidth={2.5}
                  dot={false}
                  activeDot={{ r: 4 }}
                />
              )}

              {visibleLines.pre_symptomatic && (
                <Line
                  type="monotone"
                  dataKey="pre_symptomatic"
                  name="Pre-Symptomatic Fungal"
                  stroke="#f59e0b"
                  strokeWidth={2.5}
                  strokeDasharray="4 2"
                  dot={false}
                  activeDot={{ r: 4 }}
                />
              )}

              {visibleLines.delta_pre_symp && (
                <Line
                  type="monotone"
                  dataKey="delta_pre_symp"
                  name="Early Warning Delta (Δ)"
                  stroke="#06b6d4"
                  strokeWidth={2}
                  dot={false}
                />
              )}

              {visibleLines.necrotic && (
                <Line
                  type="monotone"
                  dataKey="necrotic"
                  name="Advanced Necrotic"
                  stroke="#f43f5e"
                  strokeWidth={1.8}
                  dot={false}
                />
              )}

              {visibleLines.soil && (
                <Line
                  type="monotone"
                  dataKey="soil"
                  name="Soil Background"
                  stroke="#78716c"
                  strokeWidth={1.5}
                  dot={false}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Legend Footer Annotations */}
      <div className="grid grid-cols-4 gap-2 px-4 py-3 bg-slate-950/80 border-t border-slate-800 text-[11px] text-slate-400">
        <div>
          <span className="font-semibold text-sky-400">VIS (400-680 nm):</span> Visible Pigments
        </div>
        <div>
          <span className="font-semibold text-emerald-400">Red-Edge (700-740 nm):</span> Biomass Transition
        </div>
        <div>
          <span className="font-semibold text-amber-400">NIR (750-1300 nm):</span> Mesophyll Air Pockets
        </div>
        <div>
          <span className="font-semibold text-cyan-400">SWIR (1450 & 1940 nm):</span> Liquid Water
        </div>
      </div>
    </div>
  );
}

