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
    <div className="space-y-8 py-2">
      {/* Title */}
      <div className="border-b border-[#E5E2DC] pb-5">
        <h1 className="text-2xl font-bold text-[#171717] tracking-tight">
          Empirical Experiment Ledger
        </h1>
        <p className="text-xs text-[#66645F] mt-1">
          Historical records of training runs, hyperparameter sweeps, transfer learning freeze/unfreeze phases, and evaluation results loaded directly from <code className="bg-[#FAF9F6] border border-[#E5E2DC] px-1 py-0.5 rounded font-mono text-[11px] text-[#171717]">experiments/experiment_log.csv</code>.
        </p>
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-white p-4 rounded-xl border border-[#E5E2DC] shadow-xs flex flex-col sm:flex-row items-center justify-between gap-3">
        <div className="flex items-center gap-2 w-full sm:w-auto">
          <Filter className="w-4 h-4 text-[#8F8C85]" />
          <span className="text-xs font-medium text-[#66645F]">Filter Architecture:</span>
          <select
            value={selectedFilter}
            onChange={(e) => setSelectedFilter(e.target.value)}
            className="text-xs bg-[#FAF9F6] border border-[#E5E2DC] rounded px-2.5 py-1 text-[#171717] font-medium focus:outline-none focus:border-[#C94A3A]"
          >
            <option value="all">All Architectures ({experiments.length})</option>
            <option value="custom_cnn">Custom CNN</option>
            <option value="mobilenet_v2">MobileNetV2</option>
            <option value="resnet18">ResNet18</option>
          </select>
        </div>

        <div className="relative w-full sm:w-72">
          <Search className="w-4 h-4 text-[#8F8C85] absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search experiments or notes..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-9 pr-3 py-1.5 text-xs rounded-lg border border-[#E5E2DC] focus:outline-none focus:border-[#C94A3A] bg-[#FAF9F6] text-[#171717]"
          >
          </input>
        </div>
      </div>

      {/* Experiments Table */}
      <div className="bg-white rounded-xl border border-[#E5E2DC] shadow-xs overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-[#FAF9F6] border-b border-[#E5E2DC] text-[#66645F] font-semibold uppercase tracking-wider text-[11px]">
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
            <tbody className="divide-y divide-[#E5E2DC] text-[#171717] font-mono">
              {filtered.map((exp, idx) => (
                <tr key={`${exp.exp_id}-${idx}`} className="hover:bg-[#FAF9F6] transition-colors">
                  <td className="py-3 px-4 font-bold text-[#171717] font-sans">
                    {exp.exp_id}
                  </td>
                  <td className="py-3 px-4 font-medium text-[#66645F]">
                    {exp.model_name}
                  </td>
                  <td className="py-3 px-4 text-[11px] text-[#66645F] font-sans max-w-[140px] truncate" title={exp.augmentation}>
                    {exp.augmentation}
                  </td>
                  <td className="py-3 px-4 text-right text-[#66645F]">
                    {exp.lr} / {exp.batch_size} / {exp.epochs}
                  </td>
                  <td className="py-3 px-4 text-right text-[#66645F]">
                    {exp.val_acc}%
                  </td>
                  <td className="py-3 px-4 text-right font-bold text-[#171717]">
                    {exp.test_acc}%
                  </td>
                  <td className="py-3 px-4 text-right">
                    {exp.macro_f1}
                  </td>
                  <td className="py-3 px-4 text-right text-[#66645F]">
                    {exp.inference_time_ms} ms
                  </td>
                  <td className="py-3 px-4 font-sans text-[#66645F] text-[11px] max-w-[260px]">
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
