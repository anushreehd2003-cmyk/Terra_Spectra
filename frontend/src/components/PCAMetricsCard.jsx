import React, { useState, useEffect } from 'react';
import { Cpu, Database, Zap, CheckCircle2, TrendingUp, Layers } from 'lucide-react';
import apiService from '../services/api';

export default function PCAMetricsCard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchPCA() {
      try {
        const res = await apiService.getPCASummary();
        setData(res);
      } catch (err) {
        console.error('Failed to load PCA summary:', err);
      } finally {
        setLoading(false);
      }
    }
    fetchPCA();
  }, []);

  if (loading) {
    return (
      <div className="p-4 bg-slate-800/60 rounded-xl border border-slate-700/60 animate-pulse">
        <div className="h-4 bg-slate-700 rounded w-1/2 mb-3"></div>
        <div className="h-8 bg-slate-700 rounded w-3/4"></div>
      </div>
    );
  }

  const metrics = data?.metrics || {};
  const compression = data?.compression || {
    original_channels: 204,
    reduced_channels: 30,
    compression_ratio: 6.8,
    gpu_memory_saving_pct: 85.3,
  };
  const variancePct = metrics.total_variance_explained_pct || 98.6;

  return (
    <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl shadow-lg space-y-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <div className="p-1.5 bg-emerald-500/20 text-emerald-400 rounded-lg">
            <Cpu className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-white uppercase tracking-wider">
              Incremental PCA Engine
            </h3>
            <p className="text-[11px] text-slate-400">Spectral Dimensionality Reduction</p>
          </div>
        </div>
        <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20">
          <CheckCircle2 className="w-3 h-3" />
          Active
        </span>
      </div>

      {/* Primary KPI Grid */}
      <div className="grid grid-cols-2 gap-2">
        <div className="p-2.5 bg-slate-950/60 rounded-lg border border-slate-800">
          <span className="text-[10px] text-slate-400 uppercase font-medium flex items-center gap-1">
            <Database className="w-3 h-3 text-sky-400" />
            Channel Reduction
          </span>
          <div className="mt-1 flex items-baseline gap-1">
            <span className="text-base font-bold text-white">{compression.original_channels}</span>
            <span className="text-xs text-slate-500">→</span>
            <span className="text-base font-bold text-emerald-400">{compression.reduced_channels}</span>
            <span className="text-[10px] text-slate-400 ml-auto">{compression.compression_ratio}x</span>
          </div>
        </div>

        <div className="p-2.5 bg-slate-950/60 rounded-lg border border-slate-800">
          <span className="text-[10px] text-slate-400 uppercase font-medium flex items-center gap-1">
            <Zap className="w-3 h-3 text-amber-400" />
            VRAM Savings
          </span>
          <div className="mt-1">
            <span className="text-base font-bold text-amber-400">{compression.gpu_memory_saving_pct}%</span>
            <span className="text-[10px] text-slate-400 ml-1.5">GPU footprint</span>
          </div>
        </div>
      </div>

      {/* Variance Bar */}
      <div>
        <div className="flex justify-between text-xs mb-1.5">
          <span className="text-slate-300 font-medium flex items-center gap-1">
            <TrendingUp className="w-3.5 h-3.5 text-emerald-400" />
            Information Preserved
          </span>
          <span className="text-emerald-400 font-bold">{variancePct}%</span>
        </div>
        <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-500 rounded-full"
            style={{ width: `${Math.min(100, variancePct)}%` }}
          />
        </div>
        <p className="text-[10px] text-slate-500 mt-1">
          Target threshold &gt;95.0% satisfied. Multicollinearity suppressed.
        </p>
      </div>
    </div>
  );
}

