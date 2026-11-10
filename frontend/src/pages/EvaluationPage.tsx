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
    <div className="space-y-8 py-2">
      {/* Page Title */}
      <div className="border-b border-[#E5E2DC] pb-5">
        <h1 className="text-2xl font-bold text-[#171717] tracking-tight">
          Model Evaluation &amp; Benchmark Analysis
        </h1>
        <p className="text-xs text-[#66645F] mt-1">
          Empirical evaluation results across convolutional baseline and transfer learning architectures on the held-out GTSRB test split ($N = 3,996$ test samples).
        </p>
      </div>

      {/* Quantitative Benchmark Summary Table */}
      <div className="bg-white rounded-xl border border-[#E5E2DC] shadow-xs overflow-hidden">
        <div className="p-5 border-b border-[#E5E2DC] flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-sm font-semibold text-[#171717]">Quantitative Benchmark Matrix</h2>
            <p className="text-xs text-[#66645F]">Empirical metrics loaded directly from verified test logs.</p>
          </div>
          <span className="text-[11px] font-mono px-2.5 py-1 rounded bg-[#FAF9F6] text-[#66645F] border border-[#E5E2DC] self-start sm:self-auto">
            N = 3,996 Test Samples
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-[#FAF9F6] border-b border-[#E5E2DC] text-[#66645F] font-semibold uppercase tracking-wider text-[11px]">
                <th className="py-3 px-4">Architecture</th>
                <th className="py-3 px-4 text-right">Test Accuracy</th>
                <th className="py-3 px-4 text-right">Macro F1</th>
                <th className="py-3 px-4 text-right">Val Accuracy</th>
                <th className="py-3 px-4 text-right">CPU Latency</th>
                <th className="py-3 px-4 text-right">Parameters</th>
                <th className="py-3 px-4 text-right">Model Size</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#E5E2DC] text-[#171717] font-mono">
              {benchmark.map((row) => {
                const isSelected = row.model_id === selectedModelId;
                return (
                  <tr
                    key={row.model_id}
                    onClick={() => setSelectedModelId(row.model_id)}
                    className={`cursor-pointer transition-colors ${
                      isSelected ? 'bg-[#FAF0EE] font-semibold' : 'hover:bg-[#FAF9F6]'
                    }`}
                  >
                    <td className="py-3 px-4 font-sans font-medium flex items-center gap-2">
                      <span className={`w-2 h-2 rounded-full ${isSelected ? 'bg-[#C94A3A]' : 'bg-[#D0CDC5]'}`}></span>
                      <span>{row.model_name}</span>
                    </td>
                    <td className="py-3 px-4 text-right font-bold text-[#171717]">
                      {row.test_accuracy.toFixed(2)}%
                    </td>
                    <td className="py-3 px-4 text-right">{row.macro_f1.toFixed(4)}</td>
                    <td className="py-3 px-4 text-right">{row.val_accuracy.toFixed(2)}%</td>
                    <td className="py-3 px-4 text-right">{row.latency_ms} ms</td>
                    <td className="py-3 px-4 text-right text-[#66645F]">{row.params}</td>
                    <td className="py-3 px-4 text-right text-[#66645F]">{row.size_mb} MB</td>
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
        <div className="bg-white p-5 rounded-xl border border-[#E5E2DC] shadow-xs space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-[#171717]">
              Accuracy &amp; Macro F1 Comparison (%)
            </h3>
            <span className="text-[11px] font-mono text-[#8F8C85]">Higher is better</span>
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={accuracyData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E2DC" />
                <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#66645F' }} interval={0} />
                <YAxis domain={[90, 100]} tick={{ fontSize: 11, fill: '#66645F' }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#171717', color: '#FFFFFF', borderRadius: '6px', fontSize: '11px', border: 'none' }}
                />
                <Bar dataKey="Accuracy" fill="#C94A3A" radius={[3, 3, 0, 0]} name="Test Accuracy (%)" />
                <Bar dataKey="MacroF1" fill="#171717" radius={[3, 3, 0, 0]} name="Macro F1 (*100)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11px] text-[#66645F]">
            ResNet18 leads the benchmark with <strong>99.67%</strong> test accuracy, followed closely by the Custom CNN baseline at <strong>98.92%</strong>.
          </p>
        </div>

        {/* Latency vs Accuracy Tradeoff Scatter */}
        <div className="bg-white p-5 rounded-xl border border-[#E5E2DC] shadow-xs space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-[#171717]">
              Latency vs. Test Accuracy Trade-off
            </h3>
            <span className="text-[11px] font-mono text-[#8F8C85]">Top-left is optimal</span>
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 20, left: -20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E2DC" />
                <XAxis 
                  type="number" 
                  dataKey="latency" 
                  name="Latency (ms)" 
                  unit="ms" 
                  tick={{ fontSize: 11, fill: '#66645F' }} 
                  domain={[0, 35]}
                />
                <YAxis 
                  type="number" 
                  dataKey="accuracy" 
                  name="Accuracy (%)" 
                  unit="%" 
                  tick={{ fontSize: 11, fill: '#66645F' }} 
                  domain={[97, 100]}
                />
                <Tooltip
                  cursor={{ strokeDasharray: '3 3' }}
                  content={({ payload }) => {
                    if (payload && payload.length) {
                      const pt = payload[0].payload;
                      return (
                        <div className="bg-[#171717] text-white p-2.5 rounded-lg text-xs space-y-1 shadow-md font-mono">
                          <p className="font-sans font-semibold text-[#FAF9F6]">{pt.name}</p>
                          <p>Accuracy: {pt.accuracy}%</p>
                          <p>CPU Latency: {pt.latency} ms</p>
                          <p>Size: {pt.size} MB</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Scatter name="Models" data={scatterData} fill="#C94A3A" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11px] text-[#66645F]">
            SignNet-Lite achieves the lowest CPU inference latency (<strong>1.97 ms</strong>) while ResNet18 offers maximum accuracy with reasonable 10.94 ms latency.
          </p>
        </div>
      </div>

      {/* Selected Model Deep Dive: Confusion Matrix & Loss Curves */}
      {selectedBenchmark && (
        <div className="bg-white rounded-xl border border-[#E5E2DC] p-6 shadow-xs space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#E5E2DC] pb-4">
            <div>
              <span className="text-[11px] uppercase font-mono text-[#C94A3A] font-semibold">
                Inspecting Experiment Artifacts
              </span>
              <h2 className="text-lg font-bold text-[#171717]">
                {selectedBenchmark.model_name}
              </h2>
            </div>
            {/* Model switcher tabs */}
            <div className="flex flex-wrap gap-1.5">
              {benchmark.map((b) => (
                <button
                  key={b.model_id}
                  onClick={() => setSelectedModelId(b.model_id)}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                    b.model_id === selectedModelId
                      ? 'bg-[#171717] text-white shadow-xs'
                      : 'bg-[#FAF9F6] text-[#66645F] hover:bg-[#F1EFEA] border border-[#E5E2DC]'
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
                <span className="text-xs font-semibold text-[#171717]">
                  43-Class Normalized Confusion Matrix
                </span>
                <button
                  type="button"
                  onClick={() => setModalPlot(selectedBenchmark.confusion_matrix_plot)}
                  className="text-[11px] text-[#66645F] hover:text-[#171717] flex items-center gap-1 font-medium transition-colors"
                >
                  <ZoomIn className="w-3.5 h-3.5" />
                  Expand
                </button>
              </div>
              <div 
                className="aspect-square w-full rounded-lg border border-[#E5E2DC] overflow-hidden bg-[#FAF9F6] cursor-pointer group"
                onClick={() => setModalPlot(selectedBenchmark.confusion_matrix_plot)}
              >
                <img
                  src={selectedBenchmark.confusion_matrix_plot}
                  alt={`${selectedBenchmark.model_name} Confusion Matrix`}
                  className="w-full h-full object-contain group-hover:scale-[1.01] transition-transform"
                />
              </div>
              <p className="text-[11px] text-[#66645F]">
                Sharp diagonal concentration confirms robust classification across all 43 GTSRB sign categories with minimal off-diagonal confusion.
              </p>
            </div>

            {/* Training / Validation Curves Plot */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-[#171717]">
                  Training &amp; Validation Loss / Accuracy Curves
                </span>
                <button
                  type="button"
                  onClick={() => setModalPlot(selectedBenchmark.curves_plot)}
                  className="text-[11px] text-[#66645F] hover:text-[#171717] flex items-center gap-1 font-medium transition-colors"
                >
                  <ZoomIn className="w-3.5 h-3.5" />
                  Expand
                </button>
              </div>
              <div 
                className="aspect-square w-full rounded-lg border border-[#E5E2DC] overflow-hidden bg-[#FAF9F6] cursor-pointer group"
                onClick={() => setModalPlot(selectedBenchmark.curves_plot)}
              >
                <img
                  src={selectedBenchmark.curves_plot}
                  alt={`${selectedBenchmark.model_name} Learning Curves`}
                  className="w-full h-full object-contain group-hover:scale-[1.01] transition-transform"
                />
              </div>
              <p className="text-[11px] text-[#66645F]">
                Smooth loss decay and convergence within 5 epochs under stochastic augmentation (Rot 12° + ColorJitter).
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Lightbox Modal for Full Resolution Plots */}
      {modalPlot && (
        <div 
          className="fixed inset-0 z-50 bg-black/75 backdrop-blur-xs flex items-center justify-center p-4 cursor-zoom-out"
          onClick={() => setModalPlot(null)}
        >
          <div className="max-w-4xl max-h-[90vh] bg-white rounded-xl p-2 shadow-2xl overflow-hidden border border-[#E5E2DC]">
            <img src={modalPlot} alt="Plot Enlarged" className="max-h-[85vh] w-auto mx-auto object-contain" />
          </div>
        </div>
      )}
    </div>
  );
};
