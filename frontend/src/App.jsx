import React, { useState } from 'react';
import MapView from './components/MapView';
import { Sprout, BarChart3, Clock, AlertTriangle, ShieldCheck, Satellite } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');

  return (
    <div className="flex flex-col h-screen w-screen bg-slate-950 text-slate-100 overflow-hidden">
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
                v0.1.0 Beta
              </span>
            </h1>
            <p className="text-[11px] text-slate-400">Hyperspectral Crop Disease Forecasting Engine</p>
          </div>
        </div>

        {/* Status Indicators */}
        <div className="flex items-center gap-6 text-xs">
          <div className="flex items-center gap-2 text-slate-300">
            <Satellite className="w-4 h-4 text-sky-400" />
            <span>Target: <strong className="text-white">1,000-Acre Plot</strong></span>
          </div>
          <div className="flex items-center gap-2">
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
            </span>
            <span className="text-emerald-400 font-medium">Pipeline Ready</span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <div className="flex flex-1 relative overflow-hidden">
        {/* Left GIS Canvas */}
        <main className="flex-1 h-full relative">
          <MapView />
        </main>

        {/* Right Scaffolding Sidebar */}
        <aside className="w-80 border-l border-slate-800 bg-slate-900/95 flex flex-col p-4 gap-4 z-10">
          <div className="border-b border-slate-800 pb-3">
            <h2 className="text-sm font-semibold text-white flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-emerald-400" />
              Agricultural Analytics
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">Real-time spectral anomaly detection</p>
          </div>

          {/* Quick Metrics Placeholder */}
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-slate-800/60 rounded-lg border border-slate-700/60">
              <p className="text-[11px] text-slate-400">Total Monitored</p>
              <p className="text-lg font-bold text-white mt-1">1,000 <span className="text-xs font-normal text-slate-400">Acres</span></p>
            </div>
            <div className="p-3 bg-slate-800/60 rounded-lg border border-slate-700/60">
              <p className="text-[11px] text-slate-400">Predicted Risk</p>
              <p className="text-lg font-bold text-emerald-400 mt-1">0.0 <span className="text-xs font-normal text-slate-400">Acres</span></p>
            </div>
          </div>

          {/* Early Detection Notice */}
          <div className="p-3 bg-emerald-950/40 rounded-lg border border-emerald-800/40 text-xs text-slate-300">
            <div className="flex items-center gap-2 text-emerald-400 font-semibold mb-1">
              <ShieldCheck className="w-4 h-4" />
              <span>3-Week Pre-Symptomatic AI</span>
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              TerraSpectra monitors 200+ bands of light across 400nm-2400nm to detect subtle shifts in chlorophyll absorption before visual foliar symptoms appear.
            </p>
          </div>

          <div className="mt-auto pt-3 border-t border-slate-800 text-[11px] text-slate-500 text-center">
            Day 1: Monorepo Scaffolding & Setup Complete
          </div>
        </aside>
      </div>
    </div>
  );
}
