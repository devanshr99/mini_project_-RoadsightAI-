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
    <div className="space-y-8 py-2">
      {/* Title */}
      <div className="border-b border-[#E5E2DC] pb-5">
        <h1 className="text-2xl font-bold text-[#171717] tracking-tight">
          Neural Architecture Registry &amp; Model Specifications
        </h1>
        <p className="text-xs text-[#66645F] mt-1">
          Technical specifications for baseline convolutional networks, inverted residual bottlenecks, checkpoint state dictionaries, and Grad-CAM layer hooks.
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
                  ? 'border-[#171717] bg-white shadow-xs ring-1 ring-[#171717]'
                  : 'border-[#E5E2DC] bg-[#FAF9F6] hover:bg-white hover:border-[#D0CDC5]'
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-bold text-[#171717]">{m.display_name}</span>
                {m.available ? (
                  <span className="w-2 h-2 rounded-full bg-[#1E6B37]"></span>
                ) : (
                  <span className="w-2 h-2 rounded-full bg-[#92400E]"></span>
                )}
              </div>
              <p className="text-[11px] font-mono text-[#66645F]">{m.parameters}</p>
              <div className="mt-2 text-[11px] font-mono font-medium text-[#171717]">
                Acc: {m.test_accuracy}
              </div>
            </button>
          );
        })}
      </div>

      {/* Active Model Detailed Card */}
      {activeModel && (
        <div className="bg-white rounded-xl border border-[#E5E2DC] p-6 shadow-xs space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#E5E2DC] pb-4">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-[#FAF9F6] text-[#66645F] font-semibold border border-[#E5E2DC]">
                  {activeModel.id}
                </span>
                <span className="text-xs text-[#1E6B37] flex items-center gap-1 font-mono">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  Strict state_dict Assertion Verified
                </span>
              </div>
              <h2 className="text-xl font-bold text-[#171717]">{activeModel.display_name}</h2>
            </div>

            <div className="flex items-center gap-2 font-mono text-xs text-[#66645F] bg-[#FAF9F6] px-3 py-1.5 rounded-lg border border-[#E5E2DC]">
              <FileCheck className="w-4 h-4 text-[#8F8C85]" />
              <span>{activeModel.checkpoint}</span>
            </div>
          </div>

          <p className="text-xs text-[#66645F] leading-relaxed">
            {activeModel.description}
          </p>

          {/* Technical Specifications Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="p-3.5 rounded-lg bg-[#FAF9F6] border border-[#E5E2DC] space-y-1">
              <span className="text-[10px] uppercase font-mono text-[#8F8C85] block">Total Parameters</span>
              <strong className="text-sm font-mono text-[#171717]">{activeModel.parameters}</strong>
            </div>

            <div className="p-3.5 rounded-lg bg-[#FAF9F6] border border-[#E5E2DC] space-y-1">
              <span className="text-[10px] uppercase font-mono text-[#8F8C85] block">Checkpoint Size</span>
              <strong className="text-sm font-mono text-[#171717]">{activeModel.model_size_mb} MB</strong>
            </div>

            <div className="p-3.5 rounded-lg bg-[#FAF9F6] border border-[#E5E2DC] space-y-1">
              <span className="text-[10px] uppercase font-mono text-[#8F8C85] block">Average CPU Latency</span>
              <strong className="text-sm font-mono text-[#171717]">{activeModel.avg_cpu_latency_ms} ms</strong>
            </div>

            <div className="p-3.5 rounded-lg bg-[#FAF9F6] border border-[#E5E2DC] space-y-1">
              <span className="text-[10px] uppercase font-mono text-[#8F8C85] block">Macro F1 Score</span>
              <strong className="text-sm font-mono text-[#171717]">{activeModel.macro_f1}</strong>
            </div>
          </div>

          {/* Deep Learning Pipeline & Explainability Spec */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
            {/* Preprocessing & Tensor Pipeline */}
            <div className="space-y-3 p-4 rounded-lg border border-[#E5E2DC] bg-[#FAF9F6]">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-[#171717] flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-[#66645F]" />
                Standardized Preprocessing Pipeline
              </h3>
              <ul className="text-xs text-[#66645F] space-y-2 list-disc list-inside">
                <li>
                  <strong className="text-[#171717]">Input Normalization:</strong> Mean = [0.3337, 0.3064, 0.3171], Std = [0.2672, 0.2564, 0.2629]
                </li>
                <li>
                  <strong className="text-[#171717]">Resolution:</strong> Strict resize to exactly <strong>64×64 pixels</strong> via bilinear interpolation
                </li>
                <li>
                  <strong className="text-[#171717]">Channels:</strong> Safe conversion to 3-channel RGB (handles grayscale or alpha masks)
                </li>
                <li>
                  <strong className="text-[#171717]">Shape Assertion:</strong> Verified tensor dimension strictly enforced as <code className="bg-[#E5E2DC] px-1 py-0.5 rounded font-mono text-[11px] text-[#171717]">[1, 3, 64, 64]</code>
                </li>
              </ul>
            </div>

            {/* Grad-CAM Target Layer & Hook Mechanism */}
            <div className="space-y-3 p-4 rounded-lg border border-[#E5E2DC] bg-[#FAF9F6]">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-[#171717] flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-[#C94A3A]" />
                Grad-CAM Activation Hook Target
              </h3>
              <ul className="text-xs text-[#66645F] space-y-2 list-disc list-inside">
                <li>
                  <strong className="text-[#171717]">Target Layer:</strong> <code className="bg-[#FAF0EE] text-[#C94A3A] border border-[#F0C4BD] px-1.5 py-0.5 rounded font-mono text-[11px]">{activeModel.gradcam_target_layer}</code>
                </li>
                <li>
                  <strong className="text-[#171717]">Hook Registration:</strong> Managed with <code className="font-mono text-[11px] text-[#171717]">register_forward_hook</code> and <code className="font-mono text-[11px] text-[#171717]">register_full_backward_hook</code>
                </li>
                <li>
                  <strong className="text-[#171717]">Safe Lifecycle:</strong> Managed via context manager to explicitly call <code className="font-mono text-[11px] text-[#171717]">handle.remove()</code>, preventing PyTorch memory leaks
                </li>
                <li>
                  <strong className="text-[#171717]">Colormap Blending:</strong> OpenCV Jet color mapping with alpha blending (<code className="font-mono text-[11px] text-[#171717]">&alpha; = 0.5</code>)
                </li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
