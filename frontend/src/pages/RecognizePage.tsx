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
    <div className="space-y-8 py-4">
      {/* Page Title & Intro */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-stone-200 pb-5">
        <div>
          <h1 className="text-2xl font-bold text-stone-900 tracking-tight">Traffic Sign Recognition Workspace</h1>
          <p className="text-xs text-stone-600 mt-1">
            Upload road sign imagery or select a benchmark sample to classify and visualize Grad-CAM activation maps.
          </p>
        </div>

        {/* Model Selector Bar */}
        <div className="flex items-center gap-2 self-start md:self-auto bg-white p-1.5 rounded-lg border border-stone-200 shadow-2xs">
          <label htmlFor="model-select" className="text-xs font-medium text-stone-600 pl-2">
            Model:
          </label>
          <select
            id="model-select"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="text-xs font-semibold bg-stone-50 border border-stone-300 rounded px-2.5 py-1.5 text-stone-800 focus:outline-none focus:ring-1 focus:ring-stone-900"
          >
            {models.map((m) => (
              <option key={m.id} value={m.id} disabled={!m.available}>
                {m.display_name} {m.available ? `(${m.test_accuracy})` : '(Missing Checkpoint)'}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Main Grid: Upload & Controls on Left, Results on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Image Input & Quick Samples (5 Cols) */}
        <div className="lg:col-span-5 space-y-6">
          {/* Upload Box */}
          <div className="bg-white rounded-xl border border-stone-200 p-5 shadow-xs space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-stone-700">
                1. Select Road Sign Image
              </span>
              {previewUrl && (
                <button
                  type="button"
                  onClick={handleClearImage}
                  className="text-xs text-rose-600 hover:text-rose-700 flex items-center gap-1 font-medium"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  Clear
                </button>
              )}
            </div>

            {previewUrl ? (
              <div className="space-y-3">
                <div className="relative w-full aspect-square max-w-[280px] mx-auto rounded-lg overflow-hidden border border-stone-300 bg-stone-900 flex items-center justify-center shadow-xs">
                  <img
                    src={previewUrl}
                    alt="Selected Sign Preview"
                    className="w-full h-full object-contain"
                  />
                  <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/60 text-[10px] text-white font-mono">
                    {selectedFile ? `${selectedFile.name} (${Math.round(selectedFile.size / 1024)} KB)` : 'Test Benchmark Sample'}
                  </div>
                </div>

                <div className="flex justify-center gap-2">
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="text-xs text-stone-600 hover:text-stone-900 px-3 py-1.5 rounded border border-stone-200 hover:bg-stone-50"
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
                    ? 'border-[#e04a2f] bg-rose-50/40'
                    : 'border-stone-300 hover:border-stone-400 bg-stone-50/50'
                }`}
              >
                <div className="w-10 h-10 rounded-full bg-white shadow-2xs border border-stone-200 flex items-center justify-center text-stone-600 mb-3">
                  <Upload className="w-5 h-5 text-stone-600" />
                </div>
                <p className="text-xs font-semibold text-stone-800">
                  Click to browse or drag &amp; drop
                </p>
                <p className="text-[11px] text-stone-500 mt-1">
                  Supported formats: PNG, JPG, JPEG, WEBP (Max 10MB)
                </p>
                <span className="mt-3 px-3 py-1 rounded bg-white text-stone-700 text-[11px] font-medium border border-stone-200 shadow-3xs">
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
              className={`w-full py-3 rounded-lg font-medium text-sm flex items-center justify-center space-x-2 transition-all shadow-xs ${
                !previewUrl || loading
                  ? 'bg-stone-200 text-stone-400 cursor-not-allowed'
                  : 'bg-stone-900 text-white hover:bg-[#e04a2f] active:scale-[0.99]'
              }`}
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Processing &amp; Computing Heatmap...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  <span>Run Recognition &amp; Grad-CAM</span>
                </>
              )}
            </button>

            {error && (
              <div className="p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 shrink-0 text-rose-600 mt-0.5" />
                <span>{error}</span>
              </div>
            )}
          </div>

          {/* Quick Benchmark Sample Gallery */}
          {gallery.length > 0 && (
            <div className="bg-white rounded-xl border border-stone-200 p-5 shadow-xs space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wider text-stone-700 flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-[#e04a2f]" />
                  Or Pick a Test Sample (1-Click)
                </span>
                <span className="text-[10px] text-stone-400">GTSRB Test Set</span>
              </div>
              <p className="text-[11px] text-stone-500">
                Click any benchmark traffic sign below to test the recognition pipeline instantly:
              </p>

              <div className="grid grid-cols-4 sm:grid-cols-6 gap-2 pt-1">
                {gallery.map((sample) => (
                  <button
                    key={sample.class_id}
                    type="button"
                    onClick={() => handleSelectSample(sample)}
                    className="group flex flex-col items-center p-1.5 rounded-lg border border-stone-200 hover:border-stone-900 hover:bg-stone-50 transition-all text-center"
                    title={`Class ${sample.class_id}: ${sample.class_name}`}
                  >
                    <div className="w-12 h-12 rounded bg-stone-900 overflow-hidden flex items-center justify-center mb-1">
                      <img
                        src={sample.image_b64}
                        alt={sample.class_name}
                        className="w-full h-full object-contain group-hover:scale-105 transition-transform"
                      />
                    </div>
                    <span className="text-[9px] font-mono text-stone-600 line-clamp-1">
                      #{sample.class_id}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Active Model Summary Card */}
          {currentModelMeta && (
            <div className="p-4 rounded-xl border border-stone-200 bg-stone-50/70 text-xs text-stone-600 space-y-1.5 font-mono">
              <div className="flex justify-between font-sans font-semibold text-stone-800">
                <span>Selected Architecture:</span>
                <span>{currentModelMeta.display_name}</span>
              </div>
              <div className="flex justify-between">
                <span>Expected Input:</span>
                <span>{currentModelMeta.input_resolution}</span>
              </div>
              <div className="flex justify-between">
                <span>Grad-CAM Target Layer:</span>
                <span className="text-[#e04a2f]">{currentModelMeta.gradcam_target_layer}</span>
              </div>
              <div className="flex justify-between">
                <span>Parameters:</span>
                <span>{currentModelMeta.parameters}</span>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Prediction Results & Explainability (7 Cols) */}
        <div className="lg:col-span-7 space-y-6">
          {prediction ? (
            <div className="bg-white rounded-xl border border-stone-200 p-6 shadow-xs space-y-6">
              {/* Header: Class Label + Confidence */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-stone-200 pb-5">
                <div>
                  <div className="flex items-center space-x-2 mb-1">
                    <span className="px-2 py-0.5 rounded bg-emerald-50 text-emerald-800 border border-emerald-200 text-[10px] font-mono uppercase tracking-wider font-semibold">
                      Recognized Sign
                    </span>
                    <span className="text-xs text-stone-400 font-mono">
                      Class ID: #{prediction.predicted_class_id}
                    </span>
                  </div>
                  <h2 className="text-xl sm:text-2xl font-bold text-stone-900 tracking-tight">
                    {prediction.predicted_class_name}
                  </h2>
                </div>

                <div className="text-right sm:border-l sm:border-stone-100 sm:pl-4">
                  <div className="text-xs text-stone-500 uppercase tracking-wider">Confidence</div>
                  <div className="text-2xl sm:text-3xl font-extrabold text-stone-900 font-mono">
                    {prediction.confidence.toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Performance Badges */}
              <div className="flex flex-wrap items-center gap-4 text-xs font-mono text-stone-600 bg-stone-50 p-3 rounded-lg border border-stone-200/80">
                <div className="flex items-center gap-1.5">
                  <Clock className="w-3.5 h-3.5 text-stone-500" />
                  <span>Latency: <strong className="text-stone-900">{prediction.inference_latency_ms} ms</strong></span>
                </div>
                <div className="flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-stone-500" />
                  <span>Model: <strong className="text-stone-900">{prediction.model_name}</strong></span>
                </div>
                <div className="flex items-center gap-1.5">
                  <CheckCircle className="w-3.5 h-3.5 text-emerald-600" />
                  <span>Input Tensor: <strong className="text-stone-900">[1, 3, 64, 64]</strong></span>
                </div>
              </div>

              {/* Top 3 Predictions Bar Chart */}
              <div className="space-y-3">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-stone-700">
                  Top-3 Predicted Classes
                </h3>
                <div className="space-y-2">
                  {prediction.top3_predictions.map((top, idx) => (
                    <div key={top.class_id} className="space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-medium text-stone-800">
                          {idx + 1}. {top.class_name}
                        </span>
                        <span className="font-mono text-stone-600 font-semibold">
                          {top.confidence.toFixed(2)}%
                        </span>
                      </div>
                      <div className="w-full h-2 rounded-full bg-stone-100 overflow-hidden border border-stone-200/60">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            idx === 0 ? 'bg-[#e04a2f]' : 'bg-stone-400'
                          }`}
                          style={{ width: `${Math.max(top.confidence, 1)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Grad-CAM Visual Explainability Slider */}
              <div className="pt-2 border-t border-stone-200">
                <ImageComparisonSlider
                  originalImage={previewUrl || ''}
                  overlayImage={prediction.gradcam_overlay_b64}
                  heatmapImage={prediction.heatmap_b64}
                />
              </div>

              {/* Actions: Download Results */}
              <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-stone-100 text-xs">
                <span className="text-stone-500">Export artifacts for reports &amp; slides:</span>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => handleDownload(prediction.gradcam_overlay_b64, `gradcam_${prediction.predicted_class_id}.png`)}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-stone-200 hover:bg-stone-50 text-stone-700 font-medium transition-colors"
                  >
                    <Download className="w-3.5 h-3.5 text-stone-500" />
                    Download Grad-CAM Overlay
                  </button>
                  {prediction.heatmap_b64 && (
                    <button
                      type="button"
                      onClick={() => handleDownload(prediction.heatmap_b64, `heatmap_${prediction.predicted_class_id}.png`)}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-stone-200 hover:bg-stone-50 text-stone-700 font-medium transition-colors"
                    >
                      <Download className="w-3.5 h-3.5 text-stone-500" />
                      Download Jet Heatmap
                    </button>
                  )}
                </div>
              </div>
            </div>
          ) : (
            /* Empty State */
            <div className="bg-white rounded-xl border border-stone-200 p-12 shadow-xs text-center flex flex-col items-center justify-center min-h-[460px] space-y-4">
              <div className="w-14 h-14 rounded-full bg-stone-100 border border-stone-200 flex items-center justify-center text-stone-400">
                <ImageIcon className="w-7 h-7" />
              </div>
              <div className="max-w-sm space-y-1">
                <h3 className="text-sm font-semibold text-stone-800">No Sign Evaluated Yet</h3>
                <p className="text-xs text-stone-500 leading-relaxed">
                  Select an image from your computer or pick one of the GTSRB benchmark samples to see the class prediction and Grad-CAM saliency map.
                </p>
              </div>
              <div className="pt-2">
                <button
                  type="button"
                  onClick={() => {
                    if (gallery.length > 0) handleSelectSample(gallery[0]);
                  }}
                  className="px-4 py-2 rounded-lg bg-stone-100 hover:bg-stone-200 text-stone-800 text-xs font-medium transition-colors border border-stone-200"
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
