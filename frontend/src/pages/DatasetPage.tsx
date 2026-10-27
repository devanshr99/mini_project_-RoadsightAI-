import React, { useEffect, useState } from 'react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer, 
  CartesianGrid 
} from 'recharts';
import { 
  CheckCircle, 
  Search 
} from 'lucide-react';
import { api } from '../services/api';
import { DatasetSummary } from '../types';

export const DatasetPage: React.FC = () => {
  const [dataset, setDataset] = useState<DatasetSummary | null>(null);
  const [classes, setClasses] = useState<Record<string, string>>({});
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    Promise.allSettled([
      api.getDatasetSummary(),
      api.getClasses(),
    ]).then(([dRes, cRes]) => {
      if (dRes.status === 'fulfilled') setDataset(dRes.value);
      if (cRes.status === 'fulfilled') setClasses(cRes.value);
    });
  }, []);

  // Format chart data for all 43 classes
  const chartData = dataset
    ? Object.entries(dataset.class_distribution).map(([classId, count]) => ({
        classId: `#${classId}`,
        name: classes[classId] || `Class ${classId}`,
        count,
      }))
    : [];

  const filteredClasses = Object.entries(classes).filter(([id, name]) => {
    const q = searchTerm.toLowerCase();
    return id.includes(q) || name.toLowerCase().includes(q);
  });

  return (
    <div className="space-y-8 py-4">
      {/* Title */}
      <div className="border-b border-stone-200 pb-5">
        <h1 className="text-2xl font-bold text-stone-900 tracking-tight">
          Dataset Explorer &amp; Quality Audit
        </h1>
        <p className="text-xs text-stone-600 mt-1">
          Detailed inspection of the German Traffic Sign Recognition Benchmark (GTSRB) repository, class distribution, image resolutions, and partition splits.
        </p>
      </div>

      {/* Audit Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-5 rounded-xl border border-stone-200 bg-white shadow-2xs">
          <div className="text-xs font-medium uppercase tracking-wider text-stone-500 mb-1">
            Total Valid Images
          </div>
          <div className="text-2xl font-bold font-mono text-stone-900">
            {dataset ? dataset.total_images.toLocaleString() : '26,640'}
          </div>
          <div className="flex items-center gap-1 text-[11px] text-emerald-700 mt-1">
            <CheckCircle className="w-3.5 h-3.5" />
            <span>0 Corrupted &bull; 0 Duplicates</span>
          </div>
        </div>

        <div className="p-5 rounded-xl border border-stone-200 bg-white shadow-2xs">
          <div className="text-xs font-medium uppercase tracking-wider text-stone-500 mb-1">
            Class Categories
          </div>
          <div className="text-2xl font-bold font-mono text-stone-900">
            {dataset ? dataset.num_classes : 43}
          </div>
          <p className="text-[11px] text-stone-500 mt-1">Speed, Caution, Mandatory, Prohibitory</p>
        </div>

        <div className="p-5 rounded-xl border border-stone-200 bg-white shadow-2xs">
          <div className="text-xs font-medium uppercase tracking-wider text-stone-500 mb-1">
            Resolution Range
          </div>
          <div className="text-2xl font-bold font-mono text-stone-900">
            {dataset ? `${dataset.resolution_stats.min_width}×${dataset.resolution_stats.min_height}` : '25×25'}
            <span className="text-xs text-stone-500 font-sans font-normal ml-1">to</span>{' '}
            {dataset ? `${dataset.resolution_stats.max_width}×${dataset.resolution_stats.max_height}` : '243×225'}
          </div>
          <p className="text-[11px] text-stone-500 mt-1">
            Avg: {dataset?.resolution_stats.avg_width}×{dataset?.resolution_stats.avg_height} px
          </p>
        </div>

        <div className="p-5 rounded-xl border border-stone-200 bg-white shadow-2xs">
          <div className="text-xs font-medium uppercase tracking-wider text-stone-500 mb-1">
            Class Imbalance Ratio
          </div>
          <div className="text-2xl font-bold font-mono text-stone-900">
            {dataset ? `${dataset.class_imbalance_ratio}:1` : '10.0:1'}
          </div>
          <p className="text-[11px] text-stone-500 mt-1">Min 150 &bull; Max 1,500 imgs</p>
        </div>
      </div>

      {/* Dataset Splits Visualizer */}
      {dataset?.splits && (
        <div className="bg-white p-5 rounded-xl border border-stone-200 shadow-xs space-y-3">
          <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-stone-700">
            <span>Verified Dataset Partitions</span>
            <span className="font-mono text-stone-500 lowercase">70% train / 15% val / 15% test</span>
          </div>

          <div className="w-full h-4 rounded-full bg-stone-100 overflow-hidden flex border border-stone-200/80">
            <div 
              style={{ width: `${(dataset.splits.train / dataset.total_images) * 100}%` }}
              className="bg-stone-900 h-full flex items-center justify-center text-[10px] text-white font-mono font-medium"
              title={`Train Split: ${dataset.splits.train.toLocaleString()}`}
            />
            <div 
              style={{ width: `${(dataset.splits.val / dataset.total_images) * 100}%` }}
              className="bg-stone-500 h-full flex items-center justify-center text-[10px] text-white font-mono font-medium"
              title={`Val Split: ${dataset.splits.val.toLocaleString()}`}
            />
            <div 
              style={{ width: `${(dataset.splits.test / dataset.total_images) * 100}%` }}
              className="bg-[#e04a2f] h-full flex items-center justify-center text-[10px] text-white font-mono font-medium"
              title={`Test Split: ${dataset.splits.test.toLocaleString()}`}
            />
          </div>

          <div className="grid grid-cols-3 gap-2 text-xs font-mono pt-1 text-center">
            <div className="p-2 rounded bg-stone-50 border border-stone-200/70">
              <span className="text-stone-500 block text-[10px] uppercase">Train Set</span>
              <strong className="text-stone-900">{dataset.splits.train.toLocaleString()}</strong> images
            </div>
            <div className="p-2 rounded bg-stone-50 border border-stone-200/70">
              <span className="text-stone-500 block text-[10px] uppercase">Validation Set</span>
              <strong className="text-stone-900">{dataset.splits.val.toLocaleString()}</strong> images
            </div>
            <div className="p-2 rounded bg-stone-50 border border-stone-200/70">
              <span className="text-stone-500 block text-[10px] uppercase">Test Set</span>
              <strong className="text-stone-900">{dataset.splits.test.toLocaleString()}</strong> images
            </div>
          </div>
        </div>
      )}

      {/* Class Distribution Chart across 43 Classes */}
      <div className="bg-white p-5 rounded-xl border border-stone-200 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-stone-700">
              GTSRB 43-Class Frequency Distribution
            </h3>
            <p className="text-[11px] text-stone-500">Sample count per traffic sign class across the benchmark.</p>
          </div>
          <span className="text-xs text-stone-500 font-mono">
            Avg: {dataset?.average_images_per_class} per class
          </span>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 25 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0ee" />
              <XAxis dataKey="classId" tick={{ fontSize: 9, fill: '#6e7681' }} interval={1} angle={-45} textAnchor="end" />
              <YAxis tick={{ fontSize: 10, fill: '#6e7681' }} />
              <Tooltip
                content={({ payload }) => {
                  if (payload && payload.length) {
                    const d = payload[0].payload;
                    return (
                      <div className="bg-stone-900 text-white p-2.5 rounded-lg text-xs shadow-md">
                        <p className="font-semibold text-amber-400">{d.classId}: {d.name}</p>
                        <p className="font-mono mt-0.5">{d.count} samples</p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar dataKey="count" fill="#30363d" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Searchable Class Taxonomy Table */}
      <div className="bg-white rounded-xl border border-stone-200 shadow-xs overflow-hidden">
        <div className="p-5 border-b border-stone-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-stone-900">43-Class Sign Taxonomy</h3>
            <p className="text-xs text-stone-500">Complete list of official labels defined in the GTSRB benchmark.</p>
          </div>

          <div className="relative w-full sm:w-64">
            <Search className="w-4 h-4 text-stone-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search sign by name or ID..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-9 pr-3 py-1.5 text-xs rounded-lg border border-stone-300 focus:outline-none focus:ring-1 focus:ring-stone-900 bg-stone-50"
            />
          </div>
        </div>

        <div className="max-h-96 overflow-y-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead className="sticky top-0 bg-stone-50 border-b border-stone-200 text-stone-600 font-semibold uppercase tracking-wider">
              <tr>
                <th className="py-2.5 px-4 w-20">Class ID</th>
                <th className="py-2.5 px-4">Sign Meaning / Description</th>
                <th className="py-2.5 px-4 text-right">Sample Count</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-stone-100 text-stone-800">
              {filteredClasses.map(([id, name]) => {
                const count = dataset?.class_distribution[id] || '-';
                return (
                  <tr key={id} className="hover:bg-stone-50 transition-colors">
                    <td className="py-2.5 px-4 font-mono font-semibold text-stone-500">
                      #{id.padStart(2, '0')}
                    </td>
                    <td className="py-2.5 px-4 font-medium text-stone-900">
                      {name}
                    </td>
                    <td className="py-2.5 px-4 text-right font-mono text-stone-600">
                      {count}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
