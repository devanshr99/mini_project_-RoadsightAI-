import React, { useEffect, useState } from 'react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer, 
  CartesianGrid,
  ScatterChart,
  Scatter
} from 'recharts';
import { 
  ZoomIn
} from 'lucide-react';
import { api } from '../services/api';
import { EvaluationData } from '../types';

export const EvaluationPage: React.FC = () => {
  const [data, setData] = useState<EvaluationData | null>(null);
  const [selectedModelId, setSelectedModelId] = useState<string>('resnet18');
  const [modalPlot, setModalPlot] = useState<string | null>(null);

  useEffect(() => {
    api.getEvaluation().then(setData).catch(() => {});
  }, []);

  const benchmark = data?.benchmark || [];
  const selectedBenchmark = benchmark.find((b) => b.model_id === selectedModelId) || benchmark[0];

  const accuracyData = benchmark.map((b) => ({
    name: b.model_name.replace(' (Phase B)', ''),
    Accuracy: b.test_accuracy,
    MacroF1: Math.round(b.macro_f1 * 1000) / 10,
    LatencyMs: b.latency_ms,
  }));

  const scatterData = benchmark.map((b) => ({
    name: b.model_name,
    latency: b.latency_ms,
    accuracy: b.test_accuracy,
    size: b.size_mb,
  }));

  return (
    <div className="space-y-8 py-4">
      {/* Page Title */}
      <div className="border-b border-stone-200 pb-5">
        <h1 className="text-2xl font-bold text-stone-900 tracking-tight">
          Model Evaluation &amp; Benchmark Analysis
        </h1>
        <p className="text-xs text-stone-600 mt-1">
          Comparative empirical performance of trained convolutional and residual architectures evaluated on the held-out GTSRB test split (3,996 samples).
        </p>
      </div>

      {/* Benchmark Summary Table */}
      <div className="bg-white rounded-xl border border-stone-200 shadow-xs overflow-hidden">
        <div className="p-5 border-b border-stone-100 flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-stone-900">Quantitative Benchmark Matrix</h2>
            <p className="text-xs text-stone-500">Real empirical metrics extracted from experiment logs.</p>
          </div>
          <span className="text-[11px] font-mono px-2.5 py-1 rounded bg-stone-100 text-stone-600 border border-stone-200">
            N = 3,996 Test Samples
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-stone-50/80 border-b border-stone-200 text-stone-600 font-semibold uppercase tracking-wider">
                <th className="py-3 px-4">Architecture</th>
                <th className="py-3 px-4 text-right">Test Accuracy</th>
                <th className="py-3 px-4 text-right">Macro F1</th>
                <th className="py-3 px-4 text-right">Val Accuracy</th>
                <th className="py-3 px-4 text-right">CPU Latency</th>
                <th className="py-3 px-4 text-right">Parameters</th>
                <th className="py-3 px-4 text-right">Model Size</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-stone-100 text-stone-800 font-mono">
              {benchmark.map((row) => {
                const isSelected = row.model_id === selectedModelId;
                return (
                  <tr
                    key={row.model_id}
                    onClick={() => setSelectedModelId(row.model_id)}
                    className={`cursor-pointer transition-colors ${
                      isSelected ? 'bg-amber-50/70 font-semibold' : 'hover:bg-stone-50'
                    }`}
                  >
                    <td className="py-3 px-4 font-sans font-medium flex items-center gap-2">
                      <span className={`w-2 h-2 rounded-full ${isSelected ? 'bg-[#e04a2f]' : 'bg-stone-300'}`}></span>
                      <span>{row.model_name}</span>
                    </td>
                    <td className="py-3 px-4 text-right font-bold text-stone-900">
                      {row.test_accuracy.toFixed(2)}%
                    </td>
                    <td className="py-3 px-4 text-right">{row.macro_f1.toFixed(4)}</td>
                    <td className="py-3 px-4 text-right">{row.val_accuracy.toFixed(2)}%</td>
                    <td className="py-3 px-4 text-right">{row.latency_ms} ms</td>
                    <td className="py-3 px-4 text-right text-stone-600">{row.params}</td>
                    <td className="py-3 px-4 text-right text-stone-600">{row.size_mb} MB</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Visual Charts: Accuracy Comparison & Latency Tradeoff */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Accuracy & Macro F1 Bar Chart */}
        <div className="bg-white p-5 rounded-xl border border-stone-200 shadow-xs space-y-4">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-stone-700">
            Model Accuracy &amp; Macro F1 Comparison (%)
          </h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={accuracyData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0ee" />
                <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#6e7681' }} interval={0} />
                <YAxis domain={[90, 100]} tick={{ fontSize: 11, fill: '#6e7681' }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1c2128', color: '#fff', borderRadius: '8px', fontSize: '11px' }}
                />
                <Bar dataKey="Accuracy" fill="#e04a2f" radius={[4, 4, 0, 0]} name="Test Accuracy (%)" />
                <Bar dataKey="MacroF1" fill="#30363d" radius={[4, 4, 0, 0]} name="Macro F1 (*100)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11px] text-stone-500">
            ResNet18 leads with <strong>99.67%</strong> test accuracy, followed closely by Custom CNN baseline at <strong>98.92%</strong>.
          </p>
        </div>

        {/* Latency vs Accuracy Tradeoff Scatter */}
        <div className="bg-white p-5 rounded-xl border border-stone-200 shadow-xs space-y-4">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-stone-700">
            Inference Latency vs. Test Accuracy Trade-off
          </h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 20, left: -20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0ee" />
                <XAxis 
                  type="number" 
                  dataKey="latency" 
                  name="Latency (ms)" 
                  unit="ms" 
                  tick={{ fontSize: 11, fill: '#6e7681' }} 
                  domain={[0, 35]}
                />
                <YAxis 
                  type="number" 
                  dataKey="accuracy" 
                  name="Accuracy (%)" 
                  unit="%" 
                  tick={{ fontSize: 11, fill: '#6e7681' }} 
                  domain={[97, 100]}
                />
                <Tooltip
                  cursor={{ strokeDasharray: '3 3' }}
                  content={({ payload }) => {
                    if (payload && payload.length) {
                      const pt = payload[0].payload;
                      return (
                        <div className="bg-stone-900 text-white p-2.5 rounded-lg text-xs space-y-1 shadow-md">
                          <p className="font-semibold text-amber-400">{pt.name}</p>
                          <p>Accuracy: {pt.accuracy}%</p>
                          <p>CPU Latency: {pt.latency} ms</p>
                          <p>Model Size: {pt.size} MB</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Scatter name="Models" data={scatterData} fill="#e04a2f" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11px] text-stone-500">
            SignNet-Lite offers the lowest CPU inference latency (<strong>1.97ms</strong>) while ResNet18 offers optimal accuracy with reasonable 10.94ms runtime.
          </p>
        </div>
      </div>

      {/* Selected Model Deep Dive: Confusion Matrix & Loss Curves */}
      {selectedBenchmark && (
        <div className="bg-white rounded-xl border border-stone-200 p-6 shadow-xs space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-stone-100 pb-4">
            <div>
              <span className="text-xs uppercase font-mono text-[#e04a2f] font-semibold">
                Inspecting Experiment Artifacts
              </span>
              <h2 className="text-lg font-bold text-stone-900">
                {selectedBenchmark.model_name}
              </h2>
            </div>
            {/* Model switcher tabs */}
            <div className="flex flex-wrap gap-1.5">
              {benchmark.map((b) => (
                <button
                  key={b.model_id}
                  onClick={() => setSelectedModelId(b.model_id)}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                    b.model_id === selectedModelId
                      ? 'bg-stone-900 text-white'
                      : 'bg-stone-100 text-stone-700 hover:bg-stone-200'
                  }`}
                >
                  {b.model_name.replace(' (Phase B)', '')}
                </button>
              ))}
            </div>
          </div>

          {/* Plots Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Confusion Matrix Plot */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-stone-800">
                  43-Class Normalized Confusion Matrix
                </span>
                <button
                  type="button"
                  onClick={() => setModalPlot(selectedBenchmark.confusion_matrix_plot)}
                  className="text-[11px] text-stone-500 hover:text-stone-900 flex items-center gap-1 font-medium"
                >
                  <ZoomIn className="w-3 h-3" />
                  Expand
                </button>
              </div>
              <div 
                className="aspect-square w-full rounded-lg border border-stone-200 overflow-hidden bg-stone-50 cursor-pointer group"
                onClick={() => setModalPlot(selectedBenchmark.confusion_matrix_plot)}
              >
                <img
                  src={selectedBenchmark.confusion_matrix_plot}
                  alt={`${selectedBenchmark.model_name} Confusion Matrix`}
                  className="w-full h-full object-contain group-hover:scale-[1.01] transition-transform"
                />
              </div>
              <p className="text-[11px] text-stone-500">
                Strong diagonal concentration confirms robust classification across all 43 GTSRB sign types with minimal off-diagonal confusion.
              </p>
            </div>

            {/* Training / Validation Curves Plot */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-stone-800">
                  Training &amp; Validation Loss / Accuracy Curves
                </span>
                <button
                  type="button"
                  onClick={() => setModalPlot(selectedBenchmark.curves_plot)}
                  className="text-[11px] text-stone-500 hover:text-stone-900 flex items-center gap-1 font-medium"
                >
                  <ZoomIn className="w-3 h-3" />
                  Expand
                </button>
              </div>
              <div 
                className="aspect-square w-full rounded-lg border border-stone-200 overflow-hidden bg-stone-50 cursor-pointer group"
                onClick={() => setModalPlot(selectedBenchmark.curves_plot)}
              >
                <img
                  src={selectedBenchmark.curves_plot}
                  alt={`${selectedBenchmark.model_name} Learning Curves`}
                  className="w-full h-full object-contain group-hover:scale-[1.01] transition-transform"
                />
              </div>
              <p className="text-[11px] text-stone-500">
                Smooth loss decay and convergence within 5 epochs with data augmentation (Rot 12° + ColorJitter).
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Lightbox Modal for Full Resolution Plots */}
      {modalPlot && (
        <div 
          className="fixed inset-0 z-50 bg-black/70 backdrop-blur-xs flex items-center justify-center p-4 cursor-zoom-out"
          onClick={() => setModalPlot(null)}
        >
          <div className="max-w-4xl max-h-[90vh] bg-white rounded-xl p-2 shadow-2xl overflow-hidden">
            <img src={modalPlot} alt="Plot Enlarged" className="max-h-[85vh] w-auto mx-auto object-contain" />
          </div>
        </div>
      )}
    </div>
  );
};
