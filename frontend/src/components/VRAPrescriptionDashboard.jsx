/**
 * VRAPrescriptionDashboard.jsx
 * Variable Rate Application (VRA) Prescription Map & Economic ROI Dashboard
 *
 * Displays TerraSpectra's pre-symptomatic fungicide prescription zones with
 * per-zone dosage rates, economic savings, and groundwater reduction metrics.
 */

import { useEffect, useState } from "react";
import { apiService } from "../services/api";

// Zone colour scheme ─ mirrors GeoTIFF band-3 zone IDs (0–3)
const ZONE_CONFIG = [
  { id: 0, label: "Zone 0 – Healthy",           color: "bg-emerald-100 border-emerald-400 text-emerald-900",  dot: "bg-emerald-500",  badge: "text-emerald-700 bg-emerald-50 ring-emerald-600/20" },
  { id: 1, label: "Zone 1 – Low Risk",           color: "bg-cyan-100    border-cyan-400    text-cyan-900",     dot: "bg-cyan-500",     badge: "text-cyan-700    bg-cyan-50    ring-cyan-600/20" },
  { id: 2, label: "Zone 2 – Moderate Threat",   color: "bg-amber-100   border-amber-400   text-amber-900",    dot: "bg-amber-500",    badge: "text-amber-700   bg-amber-50   ring-amber-600/20" },
  { id: 3, label: "Zone 3 – Pre-Symptomatic",   color: "bg-red-100     border-red-400     text-red-900",      dot: "bg-red-500",      badge: "text-red-700     bg-red-50     ring-red-600/20" },
];

function StatCard({ title, value, sub, accent }) {
  return (
    <div className={`rounded-xl border p-4 flex flex-col gap-1 ${accent}`}>
      <p className="text-xs font-semibold uppercase tracking-wide opacity-70">{title}</p>
      <p className="text-2xl font-bold">{value}</p>
      {sub && <p className="text-xs opacity-60">{sub}</p>}
    </div>
  );
}

function ZoneCard({ zone, data }) {
  const cfg = ZONE_CONFIG[zone.zone_id] ?? ZONE_CONFIG[0];
  return (
    <div className={`rounded-xl border-2 p-4 ${cfg.color}`}>
      <div className="flex items-center gap-2 mb-3">
        <span className={`w-3 h-3 rounded-full ${cfg.dot}`} />
        <span className="font-semibold text-sm">{cfg.label}</span>
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
        <span className="opacity-60">Acres</span>
        <span className="font-mono font-semibold">{zone.acres?.toFixed(1) ?? "—"}</span>
        <span className="opacity-60">% Farm</span>
        <span className="font-mono font-semibold">{zone.pct_farm?.toFixed(1) ?? "—"}%</span>
        <span className="opacity-60">Dosage</span>
        <span className="font-mono font-semibold">{zone.chemical_dosage_l_per_ha?.toFixed(1) ?? "—"} L/ha</span>
        <span className="opacity-60">Priority</span>
        <span className={`inline-flex items-center rounded-md px-1.5 py-0.5 text-xs font-medium ring-1 ring-inset ${cfg.badge}`}>
          {zone.treatment_priority ?? "—"}
        </span>
      </div>
    </div>
  );
}

export default function VRAPrescriptionDashboard() {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);

  useEffect(() => {
    apiService.getPrescriptionSummary()
      .then((res) => { setData(res.data); setLoading(false); })
      .catch((err) => { setError(err.message); setLoading(false); });
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-400">
        <svg className="animate-spin h-8 w-8 mr-3 text-emerald-500" viewBox="0 0 24 24" fill="none">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
        </svg>
        Computing VRA prescription zones…
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl bg-red-50 border border-red-200 p-6 text-red-700 text-sm">
        <strong>Error loading prescription data:</strong> {error}
      </div>
    );
  }

  const eco   = data?.economic_impact ?? {};
  const zones = data?.zones ?? [];

  return (
    <div className="space-y-6 p-1">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-bold text-slate-900">Variable Rate Application Prescription</h2>
          <p className="text-sm text-slate-500 mt-0.5">
            {data?.farm_name} · {data?.total_monitored_acres?.toFixed(0)} acres monitored
          </p>
        </div>
        <div className="flex items-center gap-2 rounded-full bg-red-50 border border-red-200 px-4 py-1.5 text-sm font-semibold text-red-700">
          🔴 Pre-Symptomatic Hotspot: {data?.presymptomatic_hotspot_acres?.toFixed(1)} acres
        </div>
      </div>

      {/* Economic ROI cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          title="Blanket Spray Cost"
          value={`$${eco.blanket_spray_cost_usd?.toLocaleString() ?? "—"}`}
          sub="Uniform field treatment"
          accent="bg-slate-50 border-slate-200 text-slate-800"
        />
        <StatCard
          title="Targeted Spray Cost"
          value={`$${eco.targeted_spray_cost_usd?.toLocaleString() ?? "—"}`}
          sub="TerraSpectra VRA zones"
          accent="bg-emerald-50 border-emerald-200 text-emerald-800"
        />
        <StatCard
          title="Cost Savings"
          value={`$${eco.cost_saved_usd?.toLocaleString() ?? "—"}`}
          sub={`${eco.savings_pct?.toFixed(1) ?? "—"}% reduction`}
          accent="bg-cyan-50 border-cyan-200 text-cyan-800"
        />
        <StatCard
          title="Lead Time"
          value="21 Days"
          sub="Before visible foliar symptoms"
          accent="bg-amber-50 border-amber-200 text-amber-800"
        />
      </div>

      {/* Zone grid */}
      <div>
        <h3 className="text-sm font-semibold text-slate-700 mb-3 uppercase tracking-wide">Treatment Zones</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {zones.map((z) => (
            <ZoneCard key={z.zone_id} zone={z} />
          ))}
        </div>
      </div>

      {/* Dosage rate table */}
      <div>
        <h3 className="text-sm font-semibold text-slate-700 mb-3 uppercase tracking-wide">Per-Zone Dosage Summary</h3>
        <div className="overflow-x-auto rounded-xl border border-slate-200">
          <table className="min-w-full divide-y divide-slate-200 text-sm">
            <thead className="bg-slate-50">
              <tr>
                {["Zone", "Description", "Acres", "% Farm", "L/ha Dosage", "Priority"].map((h) => (
                  <th key={h} className="px-4 py-3 text-left text-xs font-semibold text-slate-500 uppercase tracking-wide">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 bg-white">
              {zones.map((z) => {
                const cfg = ZONE_CONFIG[z.zone_id] ?? ZONE_CONFIG[0];
                return (
                  <tr key={z.zone_id} className="hover:bg-slate-50 transition-colors">
                    <td className="px-4 py-3">
                      <span className={`inline-flex items-center gap-1.5`}>
                        <span className={`w-2.5 h-2.5 rounded-full ${cfg.dot}`}/>
                        {z.zone_id}
                      </span>
                    </td>
                    <td className="px-4 py-3 font-medium text-slate-700">{cfg.label.split("–")[1]?.trim()}</td>
                    <td className="px-4 py-3 font-mono">{z.acres?.toFixed(1)}</td>
                    <td className="px-4 py-3 font-mono">{z.pct_farm?.toFixed(1)}%</td>
                    <td className="px-4 py-3 font-mono">{z.chemical_dosage_l_per_ha?.toFixed(1)}</td>
                    <td className="px-4 py-3">
                      <span className={`inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${cfg.badge}`}>
                        {z.treatment_priority}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Footer info */}
      <p className="text-xs text-slate-400 text-center">
        Output format: GeoTIFF EPSG:4326 (WGS84) · Guidance compatible with John Deere Operations Centre &amp; Climate FieldView
      </p>
    </div>
  );
}

