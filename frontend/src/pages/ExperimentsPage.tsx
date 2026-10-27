import React, { useEffect, useState } from 'react';
import { 
  Search, 
  Filter 
} from 'lucide-react';
import { api } from '../services/api';
import { ExperimentRecord } from '../types';

export const ExperimentsPage: React.FC = () => {
  const [experiments, setExperiments] = useState<ExperimentRecord[]>([]);
  const [selectedFilter, setSelectedFilter] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    api.getExperiments()
      .then((data) => {
        setExperiments(data);
      })
      .catch(() => {});
  }, []);

  const filtered = experiments.filter((exp) => {
    const matchesModel =
      selectedFilter === 'all' || exp.model_name.toLowerCase().includes(selectedFilter.toLowerCase());
    const q = searchTerm.toLowerCase();
    const matchesSearch =
      exp.exp_id.toLowerCase().includes(q) ||
      exp.notes.toLowerCase().includes(q) ||
      exp.model_name.toLowerCase().includes(q);
    return matchesModel && matchesSearch;
  });

  return (
    <div className="space-y-8 py-4">
      {/* Title */}
      <div className="border-b border-stone-200 pb-5">
        <h1 className="text-2xl font-bold text-stone-900 tracking-tight">
          Empirical Experiment Ledger
        </h1>
        <p className="text-xs text-stone-600 mt-1">
          Historical records of training iterations, hyperparameter sweeps, transfer learning freeze/unfreeze phases, and evaluation results loaded directly from <code>experiments/experiment_log.csv</code>.
        </p>
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-white p-4 rounded-xl border border-stone-200 shadow-xs flex flex-col sm:flex-row items-center justify-between gap-3">
        <div className="flex items-center gap-2 w-full sm:w-auto">
          <Filter className="w-4 h-4 text-stone-500" />
          <span className="text-xs font-medium text-stone-600">Filter Architecture:</span>
          <select
            value={selectedFilter}
            onChange={(e) => setSelectedFilter(e.target.value)}
            className="text-xs bg-stone-50 border border-stone-300 rounded px-2.5 py-1 text-stone-800 font-medium"
          >
            <option value="all">All Architectures ({experiments.length})</option>
            <option value="custom_cnn">Custom CNN</option>
            <option value="mobilenet_v2">MobileNetV2</option>
            <option value="resnet18">ResNet18</option>
          </select>
        </div>

        <div className="relative w-full sm:w-72">
          <Search className="w-4 h-4 text-stone-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search experiments or notes..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-9 pr-3 py-1.5 text-xs rounded-lg border border-stone-300 focus:outline-none focus:ring-1 focus:ring-stone-900 bg-stone-50"
          />
        </div>
      </div>

      {/* Experiments Table */}
      <div className="bg-white rounded-xl border border-stone-200 shadow-xs overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-stone-50/80 border-b border-stone-200 text-stone-600 font-semibold uppercase tracking-wider">
                <th className="py-3 px-4">Run ID</th>
                <th className="py-3 px-4">Model</th>
                <th className="py-3 px-4">Augmentation</th>
                <th className="py-3 px-4 text-right">LR / Batch / Ep</th>
                <th className="py-3 px-4 text-right">Val Acc</th>
                <th className="py-3 px-4 text-right">Test Acc</th>
                <th className="py-3 px-4 text-right">Macro F1</th>
                <th className="py-3 px-4 text-right">Latency</th>
                <th className="py-3 px-4">Experiment Notes</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-stone-100 text-stone-800 font-mono">
              {filtered.map((exp, idx) => (
                <tr key={`${exp.exp_id}-${idx}`} className="hover:bg-stone-50 transition-colors">
                  <td className="py-3 px-4 font-bold text-stone-900 font-sans">
                    {exp.exp_id}
                  </td>
                  <td className="py-3 px-4 font-medium text-stone-700">
                    {exp.model_name}
                  </td>
                  <td className="py-3 px-4 text-[11px] text-stone-500 font-sans max-w-[140px] truncate" title={exp.augmentation}>
                    {exp.augmentation}
                  </td>
                  <td className="py-3 px-4 text-right text-stone-600">
                    {exp.lr} / {exp.batch_size} / {exp.epochs}
                  </td>
                  <td className="py-3 px-4 text-right text-stone-600">
                    {exp.val_acc}%
                  </td>
                  <td className="py-3 px-4 text-right font-bold text-stone-900">
                    {exp.test_acc}%
                  </td>
                  <td className="py-3 px-4 text-right">
                    {exp.macro_f1}
                  </td>
                  <td className="py-3 px-4 text-right text-stone-600">
                    {exp.inference_time_ms} ms
                  </td>
                  <td className="py-3 px-4 font-sans text-stone-600 text-[11px] max-w-[260px]">
                    {exp.notes}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
