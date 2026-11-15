import React, { useState, useEffect } from 'react';
import {
  Clock,
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  Sparkles,
  TrendingUp,
  DollarSign,
  ChevronRight,
  RefreshCw,
  Eye,
  CheckCircle2,
} from 'lucide-react';
import apiService from '../services/api';

export default function TemporalProgressionSlider() {
  const [timeline, setTimeline] = useState([]);
  const [selectedWeek, setSelectedWeek] = useState(-3);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchTimeline() {
      try {
        const res = await apiService.getTemporalTimeline();
        if (res?.stages) {
          setTimeline(res.stages);
        }
      } catch (err) {
        console.warn('Backend timeline call failed, using fallback.');
      } finally {
        setLoading(false);
      }
    }
    fetchTimeline();
  }, []);

  const currentStage = timeline.find((s) => s.week_offset === selectedWeek) || {
    week_offset: -3,
    label: 'Week -3 (Early Detection Window)',
    days_before_symptoms: 21,
    visual_status: 'Completely Vibrant Green (0% Visible Symptoms)',
    rgb_visible_anomalies: false,
    terraspectra_ai_alert: true,
    chlorophyll_intact: true,
    mesophyll_nir_drop_pct: 20.5,
    recommended_action: 'Apply Targeted Low-Dose Bio-fungicide & Adjust Irrigation',
    treatment_cost_savings_pct: 78.0,
    risk_acres: 265.4,
    primary_mechanism: 'Pathogen hyphae colonize spongy mesophyll air cavities; thylakoids intact.',
  };

  return (
    <div className="flex flex-col h-full w-full bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-xl p-5 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-emerald-500/20 text-emerald-400 rounded-lg border border-emerald-500/30">
            <Clock className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              4-Week Temporal Disease Progression &amp; Early Forecasting
              <span className="text-[11px] font-semibold uppercase px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                21-Day Advance Warning
              </span>
            </h2>
            <p className="text-xs text-slate-400">
              Scrub the timeline to see how TerraSpectra flags internal cellular breakdown 3 weeks before foliar symptoms become visible
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded-lg border border-emerald-500/20">
            {currentStage.days_before_symptoms} Days to Visible Symptoms
          </span>
        </div>
      </div>

      {/* Interactive Timeline Stepper */}
      <div className="p-4 bg-slate-950/80 border border-slate-800 rounded-xl space-y-3">
        <div className="flex justify-between items-center text-xs text-slate-400 mb-1">
          <span className="font-semibold text-slate-200">Timeline Progression Scrub:</span>
          <span>Click milestone to inspect canopy biophysics</span>
        </div>

        {/* 4 Step Buttons */}
        <div className="grid grid-cols-4 gap-2">
          {[-3, -2, -1, 0].map((w) => {
            const isSelected = selectedWeek === w;
            const isEarlyWindow = w === -3;
            const isOutbreak = w === 0;

            return (
              <button
                key={w}
                onClick={() => setSelectedWeek(w)}
                className={`p-3 rounded-xl border text-left transition-all ${
                  isSelected
                    ? isEarlyWindow
                      ? 'bg-emerald-500/20 border-emerald-500 text-white ring-2 ring-emerald-500/30'
                      : isOutbreak
                      ? 'bg-rose-500/20 border-rose-500 text-white ring-2 ring-rose-500/30'
                      : 'bg-amber-500/20 border-amber-500 text-white ring-2 ring-amber-500/30'
                    : 'bg-slate-900/80 border-slate-800 text-slate-400 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-bold">Week {w}</span>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800">
                    {w === 0 ? 'Outbreak' : `${Math.abs(w * 7)}d early`}
                  </span>
                </div>
                <p className="text-[11px] truncate">
                  {w === -3 && 'TerraSpectra Detection'}
                  {w === -2 && 'Pathogen Spreading'}
                  {w === -1 && 'Imminent Collapse'}
                  {w === 0 && 'Visible Foliar Necrosis'}
                </p>
              </button>
            );
          })}
        </div>
      </div>

      {/* Side-by-Side Dual Vision Comparison (RGB vs TerraSpectra Hyperspectral AI) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Left: Standard Eye / RGB Satellite View */}
        <div className="p-4 bg-slate-950/70 border border-slate-800 rounded-xl space-y-3">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-2">
              <Eye className="w-4 h-4 text-sky-400" />
              Standard RGB Camera / Sentinel-2 NDVI
            </h3>
            <span
              className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                currentStage.rgb_visible_anomalies
                  ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                  : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
              }`}
            >
              {currentStage.rgb_visible_anomalies ? 'ALERT (TOO LATE)' : 'FALSE NEGATIVE (BLIND)'}
            </span>
          </div>

          <div
            className={`h-28 rounded-lg flex flex-col items-center justify-center p-4 border transition-colors ${
              currentStage.rgb_visible_anomalies
                ? 'bg-gradient-to-br from-amber-950/60 to-rose-950/60 border-rose-800/60'
                : 'bg-emerald-950/40 border-emerald-800/40'
            }`}
          >
            <p className="text-xs font-bold text-white mb-1">
              Visual Foliage Status:
            </p>
            <p className="text-[11px] text-slate-300 text-center">
              {currentStage.visual_status}
            </p>
            <span className="text-[10px] text-slate-400 mt-2">
              {currentStage.rgb_visible_anomalies
                ? 'Severe damage already established; foliar tissue yellowed/browned.'
                : 'Foliage looks 100% healthy. Chlorophyll has not yet decomposed.'}
            </span>
          </div>

          <p className="text-[11px] text-slate-400 leading-relaxed">
            Standard RGB and 10m Sentinel-2 satellites measure only broad bands. They miss internal mesophyll cellular changes completely until leaf death in Week 0.
          </p>
        </div>

        {/* Right: TerraSpectra 204-Band Hyperspectral AI View */}
        <div className="p-4 bg-slate-950/70 border border-emerald-500/40 rounded-xl space-y-3 ring-1 ring-emerald-500/20">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-emerald-400" />
              TerraSpectra 204-Band Hybrid AI
            </h3>
            <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
              PRE-SYMPTOMATIC ALERT ACTIVE
            </span>
          </div>

          <div className="h-28 rounded-lg bg-gradient-to-br from-slate-900 to-emerald-950/60 border border-emerald-500/40 flex flex-col items-center justify-center p-4">
            <p className="text-xs font-bold text-emerald-300 mb-1">
              Mesophyll Degradation Detected:
            </p>
            <p className="text-lg font-black text-white font-mono">
              -{currentStage.mesophyll_nir_drop_pct}% NIR Reflectance
            </p>
            <span className="text-[10px] text-slate-400 mt-1">
              Red-Edge Blue Shift Inflection Confirmed
            </span>
          </div>

          <p className="text-[11px] text-slate-300 leading-relaxed">
            {currentStage.primary_mechanism}
          </p>
        </div>
      </div>

      {/* Actionable Decision Card (Economic Impact & Treatment) */}
      <div className="p-4 bg-slate-950/90 border border-slate-800 rounded-xl flex flex-wrap items-center justify-between gap-4">
        <div>
          <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider">
            Agronomic Recommendation:
          </span>
          <p className="text-xs font-semibold text-white mt-0.5">
            {currentStage.recommended_action}
          </p>
          <p className="text-[11px] text-slate-400 mt-1">
            Affected Area: <strong className="text-amber-400">{currentStage.risk_acres} Acres</strong>
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-emerald-500/10 border border-emerald-500/20 rounded-xl text-right">
            <span className="text-[10px] text-slate-400 uppercase flex items-center gap-1 justify-end">
              <DollarSign className="w-3.5 h-3.5 text-emerald-400" />
              Chemical Cost Savings
            </span>
            <p className="text-lg font-bold text-emerald-400">
              {currentStage.treatment_cost_savings_pct}%
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
