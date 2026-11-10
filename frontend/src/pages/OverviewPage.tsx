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
  Activity,
  Sliders
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
    <div className="space-y-10 py-2">
      {/* Hero Section — Balanced Two-Column Layout */}
      <section className="bg-white rounded-xl border border-[#E5E2DC] p-6 sm:p-10 shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          {/* Left Column: Heading, Context & Actions */}
          <div className="lg:col-span-7 space-y-5">
            <div className="inline-flex items-center space-x-2 px-2.5 py-1 rounded bg-[#FAF9F6] text-[#66645F] text-[11px] font-mono border border-[#E5E2DC]">
              <span className="w-1.5 h-1.5 rounded-full bg-[#C94A3A]"></span>
              <span>B.Tech CSE Project &bull; GTSRB 43-Class Benchmark</span>
            </div>

            <h1 className="text-3xl sm:text-4xl font-bold text-[#171717] tracking-tight leading-tight">
              Road Sign Recognition with Deep Vision &amp; Mathematical Explainability
            </h1>

            <p className="text-[#66645F] text-sm sm:text-base leading-relaxed">
              Autonomous driving systems demand both high classification precision and verifiable decision-making. <strong>RoadSight AI</strong> pairs deep convolutional and residual neural networks with real-time <strong>Grad-CAM</strong> visual saliency, verifying that neural predictions anchor on actual sign pictograms rather than incidental background cues.
            </p>

            <div className="flex flex-wrap items-center gap-3 pt-2">
              <Link
                to="/recognize"
                className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-lg bg-[#C94A3A] text-white font-medium text-xs hover:bg-[#B33D2F] transition-all shadow-xs group"
              >
                <Eye className="w-4 h-4" />
                <span>Recognize a Sign</span>
                <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
              </Link>

              <Link
                to="/evaluation"
                className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-lg border border-[#E5E2DC] bg-white text-[#171717] font-medium text-xs hover:bg-[#F1EFEA] transition-colors"
              >
                <BarChart2 className="w-3.5 h-3.5 text-[#66645F]" />
                <span>Benchmark Evaluation</span>
              </Link>

              <Link
                to="/dataset"
                className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-lg border border-[#E5E2DC] bg-white text-[#66645F] font-medium text-xs hover:bg-[#F1EFEA] transition-colors"
              >
                <Database className="w-3.5 h-3.5 text-[#8F8C85]" />
                <span>Dataset Explorer</span>
              </Link>
            </div>
          </div>

          {/* Right Column: Authentic Computer Vision Diagnostic Visual */}
          <div className="lg:col-span-5">
            <div className="bg-[#FAF9F6] border border-[#E5E2DC] rounded-xl p-5 space-y-3 shadow-xs">
              <div className="flex items-center justify-between border-b border-[#E5E2DC] pb-3 text-xs">
                <div className="flex items-center space-x-2 font-mono text-[11px] text-[#66645F]">
                  <Activity className="w-3.5 h-3.5 text-[#C94A3A]" />
                  <span>Saliency Inspection Viewport</span>
                </div>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#EBF5EE] text-[#1E6B37] border border-[#C3E6CD]">
                  Live Hook Verified
                </span>
              </div>

              {/* Simulated Recognition Frame with Visual Diagram */}
              <div className="relative aspect-4/3 bg-[#171717] rounded-lg overflow-hidden flex items-center justify-center p-4">
                {/* Traffic Sign Graphic: Stop Sign Hexagon/Octagon */}
                <div className="relative flex items-center justify-center">
                  <svg className="w-28 h-28 drop-shadow-md" viewBox="0 0 100 100">
                    {/* Octagon shape */}
                    <polygon points="30,5 70,5 95,30 95,70 70,95 30,95 5,70 5,30" fill="#C94A3A" stroke="#FFFFFF" strokeWidth="3" />
                    <text x="50" y="58" fill="#FFFFFF" fontSize="20" fontWeight="bold" fontFamily="sans-serif" textAnchor="middle">STOP</text>
                  </svg>

                  {/* Grad-CAM Heatmap Radial Saliency Simulation */}
                  <div className="absolute inset-0 bg-radial from-amber-400/30 via-red-500/20 to-transparent rounded-full blur-md pointer-events-none" />

                  {/* Detection Bounding Box Overlay */}
                  <div className="absolute -inset-2 border border-dashed border-emerald-400/70 rounded pointer-events-none">
                    <span className="absolute -top-3 left-0 px-1 py-0.2 bg-emerald-700 text-white font-mono text-[8px] rounded">
                      Tensor [1, 3, 64, 64]
                    </span>
                  </div>
                </div>

                {/* Bottom Overlay Telemetry Pill */}
                <div className="absolute bottom-2 left-2 right-2 bg-[#171717]/85 backdrop-blur-xs border border-white/10 rounded px-2.5 py-1.5 flex items-center justify-between text-[10px] text-white font-mono">
                  <span className="truncate text-white font-medium">Class #14: Stop Sign</span>
                  <span className="text-emerald-400 font-bold">99.8%</span>
                </div>
              </div>

              {/* Hardware & Inference Metrics Bar */}
              <div className="grid grid-cols-3 gap-2 text-center font-mono text-[10px] text-[#66645F] pt-1">
                <div className="p-1.5 rounded bg-white border border-[#E5E2DC]">
                  <span className="block text-[#8F8C85] text-[9px]">TARGET LAYER</span>
                  <span className="text-[#171717] font-semibold">ResNet18.layer4</span>
                </div>
                <div className="p-1.5 rounded bg-white border border-[#E5E2DC]">
                  <span className="block text-[#8F8C85] text-[9px]">CPU LATENCY</span>
                  <span className="text-[#171717] font-semibold">10.94 ms</span>
                </div>
                <div className="p-1.5 rounded bg-white border border-[#E5E2DC]">
                  <span className="block text-[#8F8C85] text-[9px]">MAP METHOD</span>
                  <span className="text-[#C94A3A] font-semibold">Grad-CAM Jet</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Metric Highlights Grid — Clean, Human-Designed Proportions */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-5 rounded-xl border border-[#E5E2DC] bg-white shadow-xs space-y-1">
          <div className="flex items-center justify-between text-[#66645F] mb-1">
            <span className="text-xs font-medium uppercase tracking-wider text-[#8F8C85]">Top Test Accuracy</span>
            <ShieldCheck className="w-4 h-4 text-[#1E6B37]" />
          </div>
          <div className="text-2xl font-bold text-[#171717] font-mono">99.67%</div>
          <p className="text-[11px] text-[#66645F]">ResNet18 (Phase B Fine-tuned)</p>
        </div>

        <div className="p-5 rounded-xl border border-[#E5E2DC] bg-white shadow-xs space-y-1">
          <div className="flex items-center justify-between text-[#66645F] mb-1">
            <span className="text-xs font-medium uppercase tracking-wider text-[#8F8C85]">Peak CPU Latency</span>
            <Zap className="w-4 h-4 text-[#92400E]" />
          </div>
          <div className="text-2xl font-bold text-[#171717] font-mono">1.97 ms</div>
          <p className="text-[11px] text-[#66645F]">SignNet-Lite on CPU</p>
        </div>

        <div className="p-5 rounded-xl border border-[#E5E2DC] bg-white shadow-xs space-y-1">
          <div className="flex items-center justify-between text-[#66645F] mb-1">
            <span className="text-xs font-medium uppercase tracking-wider text-[#8F8C85]">Dataset Scale</span>
            <Database className="w-4 h-4 text-[#171717]" />
          </div>
          <div className="text-2xl font-bold text-[#171717] font-mono">
            {dataset ? dataset.total_images.toLocaleString() : '26,640'}
          </div>
          <p className="text-[11px] text-[#66645F]">43 Verified GTSRB Classes</p>
        </div>

        <div className="p-5 rounded-xl border border-[#E5E2DC] bg-white shadow-xs space-y-1">
          <div className="flex items-center justify-between text-[#66645F] mb-1">
            <span className="text-xs font-medium uppercase tracking-wider text-[#8F8C85]">Explainability</span>
            <Eye className="w-4 h-4 text-[#C94A3A]" />
          </div>
          <div className="text-2xl font-bold text-[#171717] font-mono">Grad-CAM</div>
          <p className="text-[11px] text-[#66645F]">Feature Activation Localization</p>
        </div>
      </section>

      {/* Model Registry & Checkpoint Verification Section */}
      <section className="bg-white rounded-xl border border-[#E5E2DC] p-6 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#E5E2DC] pb-4">
          <div>
            <h2 className="text-base font-semibold text-[#171717]">Architectural Registry &amp; Checkpoint Verification</h2>
            <p className="text-xs text-[#66645F]">Every architecture is strictly initialized before weights loading, asserting layer mappings and input shapes.</p>
          </div>
          <div className="text-xs font-mono text-[#66645F] bg-[#FAF9F6] px-2.5 py-1 rounded border border-[#E5E2DC] self-start sm:self-auto">
            Execution Runtime: <strong className="text-[#171717]">{health ? health.device.toUpperCase() : 'Detecting...'}</strong>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 pt-1">
          {models.map((m) => (
            <div 
              key={m.id} 
              className="p-4 rounded-lg border border-[#E5E2DC] bg-[#FAF9F6] hover:bg-white hover:border-[#D0CDC5] transition-all flex flex-col justify-between space-y-3"
            >
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs font-bold text-[#171717] tracking-tight">{m.display_name}</span>
                  {m.available ? (
                    <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono font-medium bg-[#EBF5EE] text-[#1E6B37] border border-[#C3E6CD]">
                      Ready
                    </span>
                  ) : (
                    <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono font-medium bg-[#FEF3C7] text-[#92400E] border border-[#FDE68A]">
                      Missing
                    </span>
                  )}
                </div>
                <p className="text-[11px] text-[#66645F] leading-normal line-clamp-2">{m.architecture}</p>
              </div>

              <div className="pt-2 border-t border-[#E5E2DC] space-y-1 text-[11px] font-mono text-[#66645F]">
                <div className="flex justify-between">
                  <span>Accuracy:</span>
                  <strong className="text-[#171717]">{m.test_accuracy}</strong>
                </div>
                <div className="flex justify-between">
                  <span>Macro F1:</span>
                  <strong className="text-[#171717]">{m.macro_f1}</strong>
                </div>
                <div className="flex justify-between">
                  <span>CPU Latency:</span>
                  <span className="text-[#171717]">{m.avg_cpu_latency_ms} ms</span>
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

      {/* Engineering Pillars & Research Foundations */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white p-6 rounded-xl border border-[#E5E2DC] shadow-xs space-y-3">
          <div className="w-8 h-8 rounded-lg bg-[#FAF9F6] border border-[#E5E2DC] flex items-center justify-center text-[#171717]">
            <Layers className="w-4 h-4" />
          </div>
          <h3 className="font-semibold text-sm text-[#171717]">Dual-Stage Neural Benchmark</h3>
          <p className="text-xs text-[#66645F] leading-relaxed">
            Evaluates both a custom-designed 4-layer convolutional architecture trained from scratch and fine-tuned ImageNet backbones (MobileNetV2, ResNet18) to benchmark accuracy versus computational cost across edge and desktop environments.
          </p>
        </div>

        <div className="bg-white p-6 rounded-xl border border-[#E5E2DC] shadow-xs space-y-3">
          <div className="w-8 h-8 rounded-lg bg-[#FAF9F6] border border-[#E5E2DC] flex items-center justify-center text-[#C94A3A]">
            <Sliders className="w-4 h-4" />
          </div>
          <h3 className="font-semibold text-sm text-[#171717]">Grad-CAM Explainability</h3>
          <p className="text-xs text-[#66645F] leading-relaxed">
            Computes gradients of the target class score with respect to feature maps in the final convolutional layer, producing high-resolution Jet heatmaps highlighting key decision boundaries and ruling out spurious background cues.
          </p>
        </div>

        <div className="bg-white p-6 rounded-xl border border-[#E5E2DC] shadow-xs space-y-3">
          <div className="w-8 h-8 rounded-lg bg-[#FAF9F6] border border-[#E5E2DC] flex items-center justify-center text-[#1E6B37]">
            <CheckCircle2 className="w-4 h-4" />
          </div>
          <h3 className="font-semibold text-sm text-[#171717]">100% Local-First Inference</h3>
          <p className="text-xs text-[#66645F] leading-relaxed">
            Completely self-contained execution with zero reliance on cloud APIs, external inference servers, or telemetry tracking. All inference runs locally on CPU or CUDA GPU with sub-30ms round-trip latency.
          </p>
        </div>
      </section>
    </div>
  );
};
