import React, { useState, useEffect, useRef } from 'react';
import { 
  Upload, 
  Image as ImageIcon, 
  Play, 
  RefreshCw, 
  Trash2, 
  CheckCircle, 
  Clock, 
  Download, 
  Layers, 
  AlertTriangle,
  Sparkles
} from 'lucide-react';
import { api } from '../services/api';
import { ModelInfo, PredictionResult, SampleGalleryItem } from '../types';
import { ImageComparisonSlider } from '../components/ImageComparisonSlider';

export const RecognizePage: React.FC = () => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('custom_cnn');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [gallery, setGallery] = useState<SampleGalleryItem[]>([]);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load models and test samples gallery on mount
  useEffect(() => {
    api.getModels().then(setModels).catch(() => {});
    api.getSamplesGallery().then(setGallery).catch(() => {});
  }, []);

  const handleFileSelect = (file: File) => {
    setError(null);
    setPrediction(null);

    // Validate type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
      setError(`Invalid file type (${file.type}). Please select a PNG, JPG, or WEBP image.`);
      return;
    }

    // Validate size (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
      setError('File size exceeds 10MB limit.');
      return;
    }

    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleSelectSample = (sample: SampleGalleryItem) => {
    setError(null);
    setPrediction(null);
    setSelectedFile(null);
    setPreviewUrl(sample.image_b64);
  };

  const handleClearImage = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setPrediction(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleRunRecognition = async () => {
    if (!previewUrl) {
      setError('Please upload an image or pick a test sample first.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      let result: PredictionResult;
      if (selectedFile) {
        const formData = new FormData();
        formData.append('image', selectedFile);
        formData.append('model', selectedModel);
        result = await api.predict(formData);
      } else {
        // Base64 sample image
        result = await api.predictBase64(previewUrl, selectedModel);
      }
      setPrediction(result);
    } catch (err: any) {
      setError(err.message || 'Recognition execution failed. Please verify the backend status.');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = (b64Data: string, filename: string) => {
    const link = document.createElement('a');
    link.href = b64Data;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const currentModelMeta = models.find((m) => m.id === selectedModel);

  return (
    <div className="space-y-8 py-2">
      {/* Page Title & Model Selector Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#E5E2DC] pb-5">
        <div>
          <h1 className="text-2xl font-bold text-[#171717] tracking-tight">Traffic Sign Recognition Workspace</h1>
          <p className="text-xs text-[#66645F] mt-1">
            Feed road sign imagery to classify across 43 GTSRB classes and compute gradient-weighted class activation heatmaps.
          </p>
        </div>

        {/* Model Selector Bar */}
        <div className="flex items-center gap-2 self-start md:self-auto bg-white p-1.5 rounded-lg border border-[#E5E2DC] shadow-xs">
          <label htmlFor="model-select" className="text-xs font-medium text-[#66645F] pl-2">
            Target Model:
          </label>
          <select
            id="model-select"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="text-xs font-semibold bg-[#FAF9F6] border border-[#E5E2DC] rounded px-2.5 py-1.5 text-[#171717] focus:outline-none focus:border-[#C94A3A]"
          >
            {models.map((m) => (
              <option key={m.id} value={m.id} disabled={!m.available}>
                {m.display_name} {m.available ? `(${m.test_accuracy})` : '(Missing)'}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Main Grid: Upload & Inputs on Left, Predictions & Saliency on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Image Input & Quick Samples (5 Cols) */}
        <div className="lg:col-span-5 space-y-6">
          {/* Upload Card */}
          <div className="bg-white rounded-xl border border-[#E5E2DC] p-5 shadow-xs space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-[#171717]">
                1. Select Road Sign Image
              </span>
              {previewUrl && (
                <button
                  type="button"
                  onClick={handleClearImage}
                  className="text-xs text-[#C94A3A] hover:text-[#B33D2F] flex items-center gap-1 font-medium transition-colors"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  Remove
                </button>
              )}
            </div>

            {previewUrl ? (
              <div className="space-y-3">
                <div className="relative w-full aspect-square max-w-[280px] mx-auto rounded-lg overflow-hidden border border-[#E5E2DC] bg-[#171717] flex items-center justify-center shadow-xs">
                  <img
                    src={previewUrl}
                    alt="Selected Sign Preview"
                    className="w-full h-full object-contain"
                  />
                  <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/70 text-[10px] text-white font-mono">
                    {selectedFile ? `${selectedFile.name} (${Math.round(selectedFile.size / 1024)} KB)` : 'Curated Test Benchmark'}
                  </div>
                </div>

                <div className="flex justify-center gap-2">
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="text-xs text-[#66645F] hover:text-[#171717] px-3 py-1.5 rounded border border-[#E5E2DC] hover:bg-[#F1EFEA] transition-colors"
                  >
                    Replace Image
                  </button>
                </div>
              </div>
            ) : (
              <div
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`w-full aspect-4/3 rounded-lg border-2 border-dashed transition-all flex flex-col items-center justify-center p-6 text-center cursor-pointer ${
                  dragActive
                    ? 'border-[#C94A3A] bg-[#FAF0EE]'
                    : 'border-[#D0CDC5] hover:border-[#8F8C85] bg-[#FAF9F6]'
                }`}
              >
                <div className="w-10 h-10 rounded-full bg-white shadow-xs border border-[#E5E2DC] flex items-center justify-center text-[#66645F] mb-3">
                  <Upload className="w-5 h-5 text-[#66645F]" />
                </div>
                <p className="text-xs font-semibold text-[#171717]">
                  Click to browse or drag &amp; drop
                </p>
                <p className="text-[11px] text-[#66645F] mt-1">
                  Supported formats: PNG, JPG, JPEG, WEBP (Max 10MB)
                </p>
                <span className="mt-3 px-3 py-1 rounded bg-white text-[#171717] text-[11px] font-medium border border-[#E5E2DC] shadow-xs">
                  Browse Files
                </span>
              </div>
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept=".png,.jpg,.jpeg,.webp,.bmp"
              className="hidden"
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  handleFileSelect(e.target.files[0]);
                }
              }}
            />

            {/* Run Button */}
            <button
              type="button"
              onClick={handleRunRecognition}
              disabled={!previewUrl || loading}
              className={`w-full py-2.5 rounded-lg font-medium text-xs flex items-center justify-center space-x-2 transition-all shadow-xs ${
                !previewUrl || loading
                  ? 'bg-[#E5E2DC] text-[#8F8C85] cursor-not-allowed'
                  : 'bg-[#C94A3A] text-white hover:bg-[#B33D2F] active:scale-[0.99]'
              }`}
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Computing Activations &amp; Heatmap...</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>Run Recognition &amp; Grad-CAM</span>
                </>
              )}
            </button>

            {error && (
              <div className="p-3 rounded-lg bg-[#FEF3C7] border border-[#FDE68A] text-[#92400E] text-xs flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 shrink-0 text-[#92400E] mt-0.5" />
                <span>{error}</span>
              </div>
            )}
          </div>

          {/* Quick Benchmark Sample Gallery */}
          {gallery.length > 0 && (
            <div className="bg-white rounded-xl border border-[#E5E2DC] p-5 shadow-xs space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wider text-[#171717] flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-[#C94A3A]" />
                  Or Pick a Benchmark Sign (1-Click)
                </span>
                <span className="text-[10px] text-[#8F8C85] font-mono">GTSRB Test Split</span>
              </div>
              <p className="text-[11px] text-[#66645F]">
                Select any verified test sign to run inference without uploading local files:
              </p>

              <div className="grid grid-cols-4 sm:grid-cols-6 gap-2 pt-1">
                {gallery.map((sample) => (
                  <button
                    key={sample.class_id}
                    type="button"
                    onClick={() => handleSelectSample(sample)}
                    className="group flex flex-col items-center p-1.5 rounded-lg border border-[#E5E2DC] hover:border-[#171717] hover:bg-[#FAF9F6] transition-all text-center"
                    title={`Class ${sample.class_id}: ${sample.class_name}`}
                  >
                    <div className="w-12 h-12 rounded bg-[#171717] overflow-hidden flex items-center justify-center mb-1">
                      <img
                        src={sample.image_b64}
                        alt={sample.class_name}
                        className="w-full h-full object-contain group-hover:scale-105 transition-transform"
                      />
                    </div>
                    <span className="text-[9px] font-mono text-[#66645F] line-clamp-1">
                      #{sample.class_id}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Active Model Summary Card */}
          {currentModelMeta && (
            <div className="p-4 rounded-xl border border-[#E5E2DC] bg-[#FAF9F6] text-xs text-[#66645F] space-y-1.5 font-mono">
              <div className="flex justify-between font-sans font-semibold text-[#171717]">
                <span>Active Architecture:</span>
                <span>{currentModelMeta.display_name}</span>
              </div>
              <div className="flex justify-between">
                <span>Input Shape Assert:</span>
                <span>{currentModelMeta.input_resolution}</span>
              </div>
              <div className="flex justify-between">
                <span>Grad-CAM Hook Layer:</span>
                <span className="text-[#C94A3A] font-semibold">{currentModelMeta.gradcam_target_layer}</span>
              </div>
              <div className="flex justify-between">
                <span>Parameter Count:</span>
                <span>{currentModelMeta.parameters}</span>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Prediction Results & Explainability (7 Cols) */}
        <div className="lg:col-span-7 space-y-6">
          {prediction ? (
            <div className="bg-white rounded-xl border border-[#E5E2DC] p-6 shadow-xs space-y-6">
              {/* Header: Class Label + Confidence */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#E5E2DC] pb-5">
                <div>
                  <div className="flex items-center space-x-2 mb-1">
                    <span className="px-2 py-0.5 rounded bg-[#EBF5EE] text-[#1E6B37] border border-[#C3E6CD] text-[10px] font-mono uppercase tracking-wider font-semibold">
                      Recognized Sign
                    </span>
                    <span className="text-xs text-[#8F8C85] font-mono">
                      Class ID: #{prediction.predicted_class_id}
                    </span>
                  </div>
                  <h2 className="text-xl sm:text-2xl font-bold text-[#171717] tracking-tight">
                    {prediction.predicted_class_name}
                  </h2>
                </div>

                <div className="text-right sm:border-l sm:border-[#E5E2DC] sm:pl-4">
                  <div className="text-[11px] text-[#8F8C85] uppercase tracking-wider font-mono">Confidence</div>
                  <div className="text-2xl sm:text-3xl font-extrabold text-[#171717] font-mono">
                    {prediction.confidence.toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Performance Badges */}
              <div className="flex flex-wrap items-center gap-4 text-xs font-mono text-[#66645F] bg-[#FAF9F6] p-3 rounded-lg border border-[#E5E2DC]">
                <div className="flex items-center gap-1.5">
                  <Clock className="w-3.5 h-3.5 text-[#66645F]" />
                  <span>Latency: <strong className="text-[#171717]">{prediction.inference_latency_ms} ms</strong></span>
                </div>
                <div className="flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-[#66645F]" />
                  <span>Model: <strong className="text-[#171717]">{prediction.model_name}</strong></span>
                </div>
                <div className="flex items-center gap-1.5">
                  <CheckCircle className="w-3.5 h-3.5 text-[#1E6B37]" />
                  <span>Tensor: <strong className="text-[#171717]">[1, 3, 64, 64]</strong></span>
                </div>
              </div>

              {/* Top 3 Predictions Bar Chart */}
              <div className="space-y-3">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-[#171717]">
                  Top-3 Predicted Classes
                </h3>
                <div className="space-y-2">
                  {prediction.top3_predictions.map((top, idx) => (
                    <div key={top.class_id} className="space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-medium text-[#171717]">
                          {idx + 1}. {top.class_name}
                        </span>
                        <span className="font-mono text-[#66645F] font-semibold">
                          {top.confidence.toFixed(2)}%
                        </span>
                      </div>
                      <div className="w-full h-2 rounded-full bg-[#FAF9F6] overflow-hidden border border-[#E5E2DC]">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            idx === 0 ? 'bg-[#C94A3A]' : 'bg-[#8F8C85]'
                          }`}
                          style={{ width: `${Math.max(top.confidence, 1)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Grad-CAM Visual Explainability Slider */}
              <div className="pt-2 border-t border-[#E5E2DC]">
                <ImageComparisonSlider
                  originalImage={previewUrl || ''}
                  overlayImage={prediction.gradcam_overlay_b64}
                  heatmapImage={prediction.heatmap_b64}
                />
              </div>

              {/* Actions: Download Results */}
              <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-[#E5E2DC] text-xs">
                <span className="text-[#66645F]">Export artifacts for documentation &amp; slides:</span>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => handleDownload(prediction.gradcam_overlay_b64, `gradcam_${prediction.predicted_class_id}.png`)}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-[#E5E2DC] hover:bg-[#FAF9F6] text-[#171717] font-medium transition-colors"
                  >
                    <Download className="w-3.5 h-3.5 text-[#66645F]" />
                    Grad-CAM Overlay
                  </button>
                  {prediction.heatmap_b64 && (
                    <button
                      type="button"
                      onClick={() => handleDownload(prediction.heatmap_b64, `heatmap_${prediction.predicted_class_id}.png`)}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-[#E5E2DC] hover:bg-[#FAF9F6] text-[#171717] font-medium transition-colors"
                    >
                      <Download className="w-3.5 h-3.5 text-[#66645F]" />
                      Jet Heatmap
                    </button>
                  )}
                </div>
              </div>
            </div>
          ) : (
            /* Empty State */
            <div className="bg-white rounded-xl border border-[#E5E2DC] p-12 shadow-xs text-center flex flex-col items-center justify-center min-h-[460px] space-y-4">
              <div className="w-12 h-12 rounded-full bg-[#FAF9F6] border border-[#E5E2DC] flex items-center justify-center text-[#8F8C85]">
                <ImageIcon className="w-6 h-6" />
              </div>
              <div className="max-w-sm space-y-1">
                <h3 className="text-sm font-semibold text-[#171717]">No Sign Evaluated Yet</h3>
                <p className="text-xs text-[#66645F] leading-relaxed">
                  Upload an image from your computer or pick one of the GTSRB benchmark samples to see classification confidences and Grad-CAM saliency heatmaps.
                </p>
              </div>
              <div className="pt-2">
                <button
                  type="button"
                  onClick={() => {
                    if (gallery.length > 0) handleSelectSample(gallery[0]);
                  }}
                  className="px-3.5 py-2 rounded-lg bg-[#FAF9F6] hover:bg-[#F1EFEA] text-[#171717] text-xs font-medium transition-colors border border-[#E5E2DC]"
                >
                  Load Sample: Stop Sign (#14)
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
