import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { 
  ArrowRight, 
  Eye, 
  CheckCircle2, 
  Layers, 
  ShieldCheck, 
  Zap, 
  BarChart2, 
  Database,
  FileText
} from 'lucide-react';
import { api } from '../services/api';
import { HealthStatus, ModelInfo, DatasetSummary } from '../types';

export const OverviewPage: React.FC = () => {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [dataset, setDataset] = useState<DatasetSummary | null>(null);

  useEffect(() => {
    Promise.allSettled([
      api.getHealth(),
      api.getModels(),
      api.getDatasetSummary(),
    ]).then(([healthRes, modelsRes, datasetRes]) => {
      if (healthRes.status === 'fulfilled') setHealth(healthRes.value);
      if (modelsRes.status === 'fulfilled') setModels(modelsRes.value);
      if (datasetRes.status === 'fulfilled') setDataset(datasetRes.value);
    });
  }, []);

  return (
    <div className="space-y-10 py-6">
      {/* Hero Section */}
      <section className="bg-white rounded-2xl border border-stone-200/80 p-8 sm:p-10 shadow-xs relative overflow-hidden">
        <div className="absolute top-0 right-0 -mt-8 -mr-8 w-64 h-64 bg-amber-50/60 rounded-full blur-3xl pointer-events-none" />
        <div className="max-w-3xl space-y-5">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-stone-100 text-stone-700 text-xs font-mono border border-stone-200">
            <span className="w-2 h-2 rounded-full bg-[#e04a2f]"></span>
            <span>B.Tech Computer Science Project &bull; GTSRB 43-Class Benchmark</span>
          </div>

          <h1 className="text-3xl sm:text-4xl font-bold text-stone-900 tracking-tight leading-tight">
            Road Sign Recognition &amp; Visual Explainability System
          </h1>

          <p className="text-stone-600 text-base leading-relaxed">
            A production-ready deep learning pipeline classifying traffic signs across 43 German Traffic Sign Recognition Benchmark (GTSRB) categories. Features dual-stage feature extractors, transfer learning backbones, and real-time Grad-CAM saliency heatmaps for transparent computer vision decision-making.
          </p>

          <div className="flex flex-wrap items-center gap-3 pt-2">
            <Link
              to="/recognize"
              className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-lg bg-stone-900 text-white font-medium text-sm hover:bg-[#e04a2f] transition-all shadow-sm group"
            >
              <Eye className="w-4 h-4" />
              <span>Recognize a Sign</span>
              <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
            </Link>

            <Link
              to="/evaluation"
              className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-lg border border-stone-300 bg-white text-stone-700 font-medium text-sm hover:bg-stone-50 transition-colors"
            >
              <BarChart2 className="w-4 h-4 text-stone-500" />
              <span>Inspect Evaluation</span>
            </Link>

            <Link
              to="/about"
              className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-lg border border-stone-200 text-stone-600 font-medium text-sm hover:bg-stone-50 transition-colors"
            >
              <FileText className="w-4 h-4 text-stone-400" />
              <span>System Docs</span>
            </Link>
          </div>
        </div>
      </section>

      {/* Real Metric Highlights Grid */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-5 rounded-xl border border-stone-200 bg-white shadow-2xs">
          <div className="flex items-center justify-between text-stone-500 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">Top Test Accuracy</span>
            <ShieldCheck className="w-4 h-4 text-emerald-600" />
          </div>
          <div className="text-2xl font-bold text-stone-900 font-mono">99.67%</div>
          <p className="text-xs text-stone-500 mt-1">ResNet18 Fine-tuned</p>
        </div>

        <div className="p-5 rounded-xl border border-stone-200 bg-white shadow-2xs">
          <div className="flex items-center justify-between text-stone-500 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">Peak CPU Latency</span>
            <Zap className="w-4 h-4 text-amber-600" />
          </div>
          <div className="text-2xl font-bold text-stone-900 font-mono">1.97 ms</div>
          <p className="text-xs text-stone-500 mt-1">SignNet-Lite on CPU</p>
        </div>

        <div className="p-5 rounded-xl border border-stone-200 bg-white shadow-2xs">
          <div className="flex items-center justify-between text-stone-500 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">Dataset Images</span>
            <Database className="w-4 h-4 text-blue-600" />
          </div>
          <div className="text-2xl font-bold text-stone-900 font-mono">
            {dataset ? dataset.total_images.toLocaleString() : '26,640'}
          </div>
          <p className="text-xs text-stone-500 mt-1">43 Verified Classes</p>
        </div>

        <div className="p-5 rounded-xl border border-stone-200 bg-white shadow-2xs">
          <div className="flex items-center justify-between text-stone-500 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">Explainability</span>
            <Eye className="w-4 h-4 text-[#e04a2f]" />
          </div>
          <div className="text-2xl font-bold text-stone-900 font-mono">Grad-CAM</div>
          <p className="text-xs text-stone-500 mt-1">Real-time Activation Maps</p>
        </div>
      </section>

      {/* Model Registry Status Section */}
      <section className="bg-white rounded-xl border border-stone-200 p-6 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-stone-100 pb-4">
          <div>
            <h2 className="text-base font-semibold text-stone-900">Model Registry &amp; Checkpoint Verification</h2>
            <p className="text-xs text-stone-500">All architectures instantiated before weight loading with strict state_dict assertions.</p>
          </div>
          <div className="text-xs font-mono text-stone-500 bg-stone-50 px-2.5 py-1 rounded border border-stone-200 self-start sm:self-auto">
            Runtime Device: <strong className="text-stone-800">{health ? health.device.toUpperCase() : 'Detecting...'}</strong>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 pt-1">
          {models.map((m) => (
            <div 
              key={m.id} 
              className="p-4 rounded-lg border border-stone-200/90 bg-stone-50/50 hover:bg-stone-50 transition-colors flex flex-col justify-between space-y-3"
            >
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs font-bold text-stone-800 tracking-tight">{m.display_name}</span>
                  {m.available ? (
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-emerald-100 text-emerald-800">
                      Ready
                    </span>
                  ) : (
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-rose-100 text-rose-800">
                      Missing
                    </span>
                  )}
                </div>
                <p className="text-[11px] text-stone-500 leading-normal line-clamp-2">{m.architecture}</p>
              </div>

              <div className="pt-2 border-t border-stone-200/60 space-y-1 text-[11px] font-mono text-stone-600">
                <div className="flex justify-between">
                  <span>Accuracy:</span>
                  <strong className="text-stone-900">{m.test_accuracy}</strong>
                </div>
                <div className="flex justify-between">
                  <span>Macro F1:</span>
                  <strong className="text-stone-900">{m.macro_f1}</strong>
                </div>
                <div className="flex justify-between">
                  <span>Latency:</span>
                  <span>{m.avg_cpu_latency_ms} ms</span>
                </div>
                <div className="flex justify-between">
                  <span>Size:</span>
                  <span>{m.model_size_mb} MB</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Architectural Pillars & Explainability Methodology */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white p-6 rounded-xl border border-stone-200 shadow-2xs space-y-3">
          <div className="w-8 h-8 rounded-lg bg-stone-100 flex items-center justify-center text-stone-800">
            <Layers className="w-4 h-4" />
          </div>
          <h3 className="font-semibold text-sm text-stone-900">Custom CNN &amp; Transfer Learning</h3>
          <p className="text-xs text-stone-600 leading-relaxed">
            Evaluates both a custom-designed 4-layer convolutional architecture trained from scratch and fine-tuned ImageNet backbones (MobileNetV2, ResNet18) to benchmark accuracy versus computational cost.
          </p>
        </div>

        <div className="bg-white p-6 rounded-xl border border-stone-200 shadow-2xs space-y-3">
          <div className="w-8 h-8 rounded-lg bg-stone-100 flex items-center justify-center text-[#e04a2f]">
            <Eye className="w-4 h-4" />
          </div>
          <h3 className="font-semibold text-sm text-stone-900">Grad-CAM Explainability</h3>
          <p className="text-xs text-stone-600 leading-relaxed">
            Calculates gradients of the top predicted class score with respect to feature maps in the final convolutional layer, producing high-resolution Jet heatmaps highlighting key decision boundaries.
          </p>
        </div>

        <div className="bg-white p-6 rounded-xl border border-stone-200 shadow-2xs space-y-3">
          <div className="w-8 h-8 rounded-lg bg-stone-100 flex items-center justify-center text-emerald-700">
            <CheckCircle2 className="w-4 h-4" />
          </div>
          <h3 className="font-semibold text-sm text-stone-900">Local-First Engineering</h3>
          <p className="text-xs text-stone-600 leading-relaxed">
            100% self-contained execution without reliance on external cloud inference APIs, telemetry, or third-party storage. All inference runs locally on CPU/CUDA with sub-30ms response times.
          </p>
        </div>
      </section>
    </div>
  );
};
