import React, { useEffect, useState } from 'react';
import { 
  Layers, 
  CheckCircle2, 
  FileCheck, 
  Sliders 
} from 'lucide-react';
import { api } from '../services/api';
import { ModelInfo } from '../types';

export const ModelsPage: React.FC = () => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('custom_cnn');

  useEffect(() => {
    api.getModels().then(setModels).catch(() => {});
  }, []);

  const activeModel = models.find((m) => m.id === selectedModel) || models[0];

  return (
    <div className="space-y-8 py-4">
      {/* Title */}
      <div className="border-b border-stone-200 pb-5">
        <h1 className="text-2xl font-bold text-stone-900 tracking-tight">
          Neural Architecture Registry &amp; Model Specifications
        </h1>
        <p className="text-xs text-stone-600 mt-1">
          Detailed inspection of convolutional layers, bottleneck residual blocks, checkpoint state dictionaries, and Grad-CAM target layer hooks.
        </p>
      </div>

      {/* Model Selection Tabs */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {models.map((m) => {
          const isSelected = m.id === selectedModel;
          return (
            <button
              key={m.id}
              onClick={() => setSelectedModel(m.id)}
              className={`p-4 rounded-xl border text-left transition-all ${
                isSelected
                  ? 'border-stone-900 bg-white shadow-xs ring-1 ring-stone-900'
                  : 'border-stone-200 bg-stone-50/60 hover:bg-white hover:border-stone-300'
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-bold text-stone-900">{m.display_name}</span>
                {m.available ? (
                  <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
                ) : (
                  <span className="w-2 h-2 rounded-full bg-rose-500"></span>
                )}
              </div>
              <p className="text-[11px] font-mono text-stone-500">{m.parameters}</p>
              <div className="mt-2 text-[11px] font-mono font-medium text-stone-800">
                Acc: {m.test_accuracy}
              </div>
            </button>
          );
        })}
      </div>

      {/* Active Model Detailed Card */}
      {activeModel && (
        <div className="bg-white rounded-xl border border-stone-200 p-6 shadow-xs space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-stone-100 pb-4">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-stone-100 text-stone-700 font-semibold border border-stone-200">
                  {activeModel.id}
                </span>
                <span className="text-xs text-emerald-700 flex items-center gap-1 font-mono">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  Checkpoint Verified (Strict State Dict)
                </span>
              </div>
              <h2 className="text-xl font-bold text-stone-900">{activeModel.display_name}</h2>
            </div>

            <div className="flex items-center gap-2 font-mono text-xs text-stone-600 bg-stone-50 px-3 py-1.5 rounded-lg border border-stone-200">
              <FileCheck className="w-4 h-4 text-stone-500" />
              <span>{activeModel.checkpoint}</span>
            </div>
          </div>

          <p className="text-xs text-stone-600 leading-relaxed">
            {activeModel.description}
          </p>

          {/* Technical Specifications Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="p-3.5 rounded-lg bg-stone-50 border border-stone-200/80 space-y-1">
              <span className="text-[10px] uppercase font-mono text-stone-500 block">Total Parameters</span>
              <strong className="text-sm font-mono text-stone-900">{activeModel.parameters}</strong>
            </div>

            <div className="p-3.5 rounded-lg bg-stone-50 border border-stone-200/80 space-y-1">
              <span className="text-[10px] uppercase font-mono text-stone-500 block">Checkpoint Size</span>
              <strong className="text-sm font-mono text-stone-900">{activeModel.model_size_mb} MB</strong>
            </div>

            <div className="p-3.5 rounded-lg bg-stone-50 border border-stone-200/80 space-y-1">
              <span className="text-[10px] uppercase font-mono text-stone-500 block">Average CPU Latency</span>
              <strong className="text-sm font-mono text-stone-900">{activeModel.avg_cpu_latency_ms} ms</strong>
            </div>

            <div className="p-3.5 rounded-lg bg-stone-50 border border-stone-200/80 space-y-1">
              <span className="text-[10px] uppercase font-mono text-stone-500 block">Macro F1 Score</span>
              <strong className="text-sm font-mono text-stone-900">{activeModel.macro_f1}</strong>
            </div>
          </div>

          {/* Deep Learning Pipeline & Explainability Spec */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
            {/* Preprocessing & Tensor Pipeline */}
            <div className="space-y-3 p-4 rounded-lg border border-stone-200 bg-stone-50/50">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-stone-800 flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-stone-600" />
                Standardized Preprocessing Pipeline
              </h3>
              <ul className="text-xs text-stone-600 space-y-2 list-disc list-inside">
                <li>
                  <strong>Input Normalization:</strong> Mean = [0.3337, 0.3064, 0.3171], Std = [0.2672, 0.2564, 0.2629]
                </li>
                <li>
                  <strong>Input Dimensions:</strong> Strict resize to exactly <strong>64×64 pixels</strong> (Bilinear interpolation)
                </li>
                <li>
                  <strong>Channels:</strong> Safe conversion to 3-channel RGB (handles single-channel grayscale or RGBA)
                </li>
                <li>
                  <strong>Shape Assertion:</strong> Verified tensor dimension strictly enforced as <code className="bg-stone-200/80 px-1 py-0.5 rounded font-mono text-[11px]">[1, 3, 64, 64]</code>
                </li>
              </ul>
            </div>

            {/* Grad-CAM Target Layer & Hook Mechanism */}
            <div className="space-y-3 p-4 rounded-lg border border-stone-200 bg-stone-50/50">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-stone-800 flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-[#e04a2f]" />
                Grad-CAM Activation Hook Target
              </h3>
              <ul className="text-xs text-stone-600 space-y-2 list-disc list-inside">
                <li>
                  <strong>Target Layer:</strong> <code className="bg-rose-50 text-[#e04a2f] border border-rose-200 px-1.5 py-0.5 rounded font-mono text-[11px]">{activeModel.gradcam_target_layer}</code>
                </li>
                <li>
                  <strong>Hook Management:</strong> Registered with <code className="font-mono text-[11px]">register_forward_hook</code> and <code className="font-mono text-[11px]">register_full_backward_hook</code>
                </li>
                <li>
                  <strong>Safe Lifecycle:</strong> Managed via context manager to explicitly call <code className="font-mono text-[11px]">handle.remove()</code> upon completion, preventing memory leaks
                </li>
                <li>
                  <strong>Colormap:</strong> OpenCV Jet color mapping with alpha blending (<code className="font-mono text-[11px]">α = 0.5</code>)
                </li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
