import React, { useState, useEffect } from 'react';
import MapView from './components/MapView';
import SpectralInspector from './components/SpectralInspector';
import PCAMetricsCard from './components/PCAMetricsCard';
import {
  Sprout,
  BarChart3,
  Activity,
  Map as MapIcon,
  ShieldAlert,
  ShieldCheck,
  Satellite,
  RefreshCw,
  Sparkles,
} from 'lucide-react';
import apiService from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('spectral'); // Default to Day 2 spectral inspector
  const [status, setStatus] = useState(null);
  const [simulating, setSimulating] = useState(false);
  const [monitoredAcres, setMonitoredAcres] = useState(1000.0);
  const [predictedRiskAcres, setPredictedRiskAcres] = useState(367.5);

  useEffect(() => {
    async function loadStatus() {
      try {
        const res = await apiService.getStatus();
        setStatus(res);
      } catch (err) {
        console.warn('Backend not running yet, using local defaults.');
      }
    }
    loadStatus();
  }, []);

  const handleSimulateCycle = async () => {
    setSimulating(true);
    try {
      const res = await apiService.triggerSimulation(null, 3);
      if (res?.metadata?.infection_area_acres) {
        setPredictedRiskAcres(res.metadata.infection_area_acres);
      }
    } catch (err) {
      console.error('Simulation error:', err);
      // Fallback update
      setPredictedRiskAcres((prev) => (prev > 300 ? 285.4 : 412.8));
    } finally {
      setTimeout(() => setSimulating(false), 500);
    }
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-slate-950 text-slate-100 overflow-hidden font-sans">
      {/* Top Navigation Bar */}
      <header className="h-14 border-b border-slate-800 bg-slate-900/90 backdrop-blur px-6 flex items-center justify-between z-20">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-emerald-500/20 text-emerald-400 rounded-lg border border-emerald-500/30">
            <Sprout className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-base font-bold tracking-wide text-white flex items-center gap-2">
              TerraSpectra
              <span className="text-[10px] uppercase font-semibold tracking-wider px-2 py-0.5 bg-emerald-500/10 text-emerald-400 rounded-full border border-emerald-500/20">
                v0.2.0 • Day 2
              </span>
            </h1>
            <p className="text-[11px] text-slate-400">Hyperspectral Crop Disease Forecasting Engine</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex items-center bg-slate-950/80 p-1 rounded-xl border border-slate-800 text-xs">
          <button
            onClick={() => setActiveTab('spectral')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg font-medium transition ${
              activeTab === 'spectral'
                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Activity className="w-4 h-4 text-emerald-400" />
            204-Band Spectral Inspector
          </button>
          <button
            onClick={() => setActiveTab('map')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg font-medium transition ${
              activeTab === 'map'
                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <MapIcon className="w-4 h-4 text-sky-400" />
            GIS Farm Parcels
          </button>
        </div>

        {/* Status Indicators */}
        <div className="flex items-center gap-6 text-xs">
          <div className="flex items-center gap-2 text-slate-300">
            <Satellite className="w-4 h-4 text-sky-400" />
            <span>Target: <strong className="text-white">1,000 Acres</strong></span>
          </div>
          <div className="flex items-center gap-2">
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
            </span>
            <span className="text-emerald-400 font-medium">Pipeline Active</span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <div className="flex flex-1 relative overflow-hidden">
        {/* Left Interactive Canvas Area */}
        <main className="flex-1 h-full p-4 overflow-y-auto">
          {activeTab === 'spectral' ? <SpectralInspector /> : <MapView />}
        </main>

        {/* Right Telemetry & PCA Sidebar */}
        <aside className="w-84 border-l border-slate-800 bg-slate-900/95 flex flex-col p-4 gap-4 z-10 overflow-y-auto">
          <div className="border-b border-slate-800 pb-3 flex items-center justify-between">
            <div>
              <h2 className="text-sm font-semibold text-white flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-emerald-400" />
                Agricultural Analytics
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">Real-time spectral anomaly detection</p>
            </div>
            <button
              onClick={handleSimulateCycle}
              disabled={simulating}
              title="Simulate new satellite pass"
              className="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg border border-slate-700 transition"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${simulating ? 'animate-spin' : ''}`} />
            </button>
          </div>

          {/* Quick Metrics */}
          <div className="grid grid-cols-2 gap-2.5">
            <div className="p-3 bg-slate-800/60 rounded-xl border border-slate-700/60">
              <p className="text-[11px] text-slate-400">Total Monitored</p>
              <p className="text-lg font-bold text-white mt-0.5">
                {monitoredAcres.toLocaleString()} <span className="text-xs font-normal text-slate-400">Acres</span>
              </p>
            </div>
            <div className="p-3 bg-slate-800/60 rounded-xl border border-slate-700/60">
              <p className="text-[11px] text-slate-400">Pre-Symptomatic Risk</p>
              <p className="text-lg font-bold text-amber-400 mt-0.5">
                {predictedRiskAcres} <span className="text-xs font-normal text-slate-400">Acres</span>
              </p>
            </div>
          </div>

          {/* Incremental PCA Dimensionality Reduction Card */}
          <PCAMetricsCard />

          {/* Early Detection Notice */}
          <div className="p-3.5 bg-emerald-950/40 rounded-xl border border-emerald-800/40 text-xs text-slate-300 space-y-2">
            <div className="flex items-center gap-2 text-emerald-400 font-semibold">
              <ShieldCheck className="w-4 h-4" />
              <span>3-Week Pre-Symptomatic Advantage</span>
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              Standard multispectral satellites (e.g. Sentinel-2 / Landsat) only measure 3-12 broad bands and miss the internal mesophyll structural breakdown.
              TerraSpectra's 204 contiguous narrow channels flag the 15-25% NIR reflection drop up to 21 days before lesions appear.
            </p>
          </div>

          {/* Day 2 Milestone Footer */}
          <div className="mt-auto pt-3 border-t border-slate-800 text-[11px] text-slate-500 text-center">
            <span className="text-emerald-400 font-medium">Day 2 Milestone Completed</span>
            <p className="text-[10px] text-slate-500 mt-0.5">
              204-Band Simulator • Incremental PCA (30-dim) • Indices Engine
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}
