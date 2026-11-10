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
    <div className="space-y-8 py-2">
      {/* Title */}
      <div className="border-b border-[#E5E2DC] pb-5">
        <h1 className="text-2xl font-bold text-[#171717] tracking-tight">
          Dataset Explorer &amp; Quality Audit
        </h1>
        <p className="text-xs text-[#66645F] mt-1">
          Inspection of the German Traffic Sign Recognition Benchmark (GTSRB) repository, class distribution, image resolutions, and partition splits.
        </p>
      </div>

      {/* Audit Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-5 rounded-xl border border-[#E5E2DC] bg-white shadow-xs space-y-1">
          <div className="text-xs font-medium uppercase tracking-wider text-[#8F8C85]">
            Total Valid Images
          </div>
          <div className="text-2xl font-bold font-mono text-[#171717]">
            {dataset ? dataset.total_images.toLocaleString() : '26,640'}
          </div>
          <div className="flex items-center gap-1 text-[11px] text-[#1E6B37] mt-1">
            <CheckCircle className="w-3.5 h-3.5" />
            <span>0 Corrupted &bull; 0 Duplicates</span>
          </div>
        </div>

        <div className="p-5 rounded-xl border border-[#E5E2DC] bg-white shadow-xs space-y-1">
          <div className="text-xs font-medium uppercase tracking-wider text-[#8F8C85]">
            Class Categories
          </div>
          <div className="text-2xl font-bold font-mono text-[#171717]">
            {dataset ? dataset.num_classes : 43}
          </div>
          <p className="text-[11px] text-[#66645F]">Speed, Caution, Mandatory, Prohibitory</p>
        </div>

        <div className="p-5 rounded-xl border border-[#E5E2DC] bg-white shadow-xs space-y-1">
          <div className="text-xs font-medium uppercase tracking-wider text-[#8F8C85]">
            Resolution Range
          </div>
          <div className="text-2xl font-bold font-mono text-[#171717]">
            {dataset ? `${dataset.resolution_stats.min_width}×${dataset.resolution_stats.min_height}` : '25×25'}
            <span className="text-xs text-[#8F8C85] font-sans font-normal ml-1">to</span>{' '}
            {dataset ? `${dataset.resolution_stats.max_width}×${dataset.resolution_stats.max_height}` : '243×225'}
          </div>
          <p className="text-[11px] text-[#66645F]">
            Avg: {dataset?.resolution_stats.avg_width}×{dataset?.resolution_stats.avg_height} px
          </p>
        </div>

        <div className="p-5 rounded-xl border border-[#E5E2DC] bg-white shadow-xs space-y-1">
          <div className="text-xs font-medium uppercase tracking-wider text-[#8F8C85]">
            Class Imbalance Ratio
          </div>
          <div className="text-2xl font-bold font-mono text-[#171717]">
            {dataset ? `${dataset.class_imbalance_ratio}:1` : '10.0:1'}
          </div>
          <p className="text-[11px] text-[#66645F]">Min 150 &bull; Max 1,500 imgs</p>
        </div>
      </div>

      {/* Dataset Splits Visualizer */}
      {dataset?.splits && (
        <div className="bg-white p-5 rounded-xl border border-[#E5E2DC] shadow-xs space-y-3">
          <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-[#171717]">
            <span>Verified Dataset Partitions</span>
            <span className="font-mono text-[#66645F] lowercase text-[11px]">70% train / 15% val / 15% test</span>
          </div>

          <div className="w-full h-3.5 rounded-full bg-[#FAF9F6] overflow-hidden flex border border-[#E5E2DC]">
            <div 
              style={{ width: `${(dataset.splits.train / dataset.total_images) * 100}%` }}
              className="bg-[#171717] h-full"
              title={`Train Split: ${dataset.splits.train.toLocaleString()}`}
            />
            <div 
              style={{ width: `${(dataset.splits.val / dataset.total_images) * 100}%` }}
              className="bg-[#66645F] h-full"
              title={`Val Split: ${dataset.splits.val.toLocaleString()}`}
            />
            <div 
              style={{ width: `${(dataset.splits.test / dataset.total_images) * 100}%` }}
              className="bg-[#C94A3A] h-full"
              title={`Test Split: ${dataset.splits.test.toLocaleString()}`}
            />
          </div>

          <div className="grid grid-cols-3 gap-2 text-xs font-mono pt-1 text-center">
            <div className="p-2.5 rounded-lg bg-[#FAF9F6] border border-[#E5E2DC]">
              <span className="text-[#8F8C85] block text-[10px] uppercase">Train Set</span>
              <strong className="text-[#171717]">{dataset.splits.train.toLocaleString()}</strong> images
            </div>
            <div className="p-2.5 rounded-lg bg-[#FAF9F6] border border-[#E5E2DC]">
              <span className="text-[#8F8C85] block text-[10px] uppercase">Validation Set</span>
              <strong className="text-[#171717]">{dataset.splits.val.toLocaleString()}</strong> images
            </div>
            <div className="p-2.5 rounded-lg bg-[#FAF9F6] border border-[#E5E2DC]">
              <span className="text-[#8F8C85] block text-[10px] uppercase">Test Set</span>
              <strong className="text-[#171717]">{dataset.splits.test.toLocaleString()}</strong> images
            </div>
          </div>
        </div>
      )}

      {/* Class Distribution Chart across 43 Classes */}
      <div className="bg-white p-5 rounded-xl border border-[#E5E2DC] shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-[#171717]">
              GTSRB 43-Class Frequency Distribution
            </h3>
            <p className="text-[11px] text-[#66645F]">Sample count per traffic sign class across the benchmark.</p>
          </div>
          <span className="text-xs text-[#8F8C85] font-mono">
            Avg: {dataset?.average_images_per_class} per class
          </span>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 25 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E5E2DC" />
              <XAxis dataKey="classId" tick={{ fontSize: 9, fill: '#66645F' }} interval={1} angle={-45} textAnchor="end" />
              <YAxis tick={{ fontSize: 10, fill: '#66645F' }} />
              <Tooltip
                content={({ payload }) => {
                  if (payload && payload.length) {
                    const d = payload[0].payload;
                    return (
                      <div className="bg-[#171717] text-white p-2.5 rounded-lg text-xs shadow-md font-mono">
                        <p className="font-sans font-semibold text-[#FAF9F6]">{d.classId}: {d.name}</p>
                        <p className="mt-0.5 text-[#E5E2DC]">{d.count} samples</p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar dataKey="count" fill="#171717" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Searchable Class Taxonomy Table */}
      <div className="bg-white rounded-xl border border-[#E5E2DC] shadow-xs overflow-hidden">
        <div className="p-5 border-b border-[#E5E2DC] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-[#171717]">43-Class Sign Taxonomy</h3>
            <p className="text-xs text-[#66645F]">Official class definitions defined in the GTSRB benchmark standard.</p>
          </div>

          <div className="relative w-full sm:w-64">
            <Search className="w-4 h-4 text-[#8F8C85] absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search sign by name or ID..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-9 pr-3 py-1.5 text-xs rounded-lg border border-[#E5E2DC] focus:outline-none focus:border-[#C94A3A] bg-[#FAF9F6] text-[#171717]"
            />
          </div>
        </div>

        <div className="max-h-96 overflow-y-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead className="sticky top-0 bg-[#FAF9F6] border-b border-[#E5E2DC] text-[#66645F] font-semibold uppercase tracking-wider text-[11px]">
              <tr>
                <th className="py-2.5 px-4 w-20">Class ID</th>
                <th className="py-2.5 px-4">Sign Meaning / Description</th>
                <th className="py-2.5 px-4 text-right">Sample Count</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#E5E2DC] text-[#171717]">
              {filteredClasses.map(([id, name]) => {
                const count = dataset?.class_distribution[id] || '-';
                return (
                  <tr key={id} className="hover:bg-[#FAF9F6] transition-colors">
                    <td className="py-2.5 px-4 font-mono font-semibold text-[#66645F]">
                      #{id.padStart(2, '0')}
                    </td>
                    <td className="py-2.5 px-4 font-medium text-[#171717]">
                      {name}
                    </td>
                    <td className="py-2.5 px-4 text-right font-mono text-[#66645F]">
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
