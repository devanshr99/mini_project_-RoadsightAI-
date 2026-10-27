export interface ModelInfo {
  id: string;
  display_name: string;
  architecture: string;
  checkpoint: string;
  parameters: string;
  model_size_mb: number;
  input_resolution: string;
  test_accuracy: string;
  macro_f1: string;
  avg_cpu_latency_ms: number;
  supports_gradcam: boolean;
  gradcam_target_layer: string;
  description: string;
  available: boolean;
  actual_file_size_mb: number;
}

export interface Top3Prediction {
  class_id: number;
  class_name: string;
  confidence: number;
}

export interface PredictionResult {
  predicted_class_id: number;
  predicted_class_name: string;
  confidence: number;
  top3_predictions: Top3Prediction[];
  gradcam_overlay_b64: string;
  heatmap_b64: string;
  model_name: string;
  checkpoint_loaded: string;
  inference_latency_ms: number;
  input_dimensions: string;
  is_valid: boolean;
}

export interface HealthStatus {
  status: string;
  timestamp: string;
  torch_version: string;
  device: string;
  cuda_available: boolean;
  models: Record<string, {
    checkpoint_path: string;
    available: boolean;
    size_mb: number;
  }>;
  dataset_ready: boolean;
  server: string;
}

export interface DatasetSummary {
  dataset_name: string;
  num_classes: number;
  total_images: number;
  valid_images: number;
  corrupted_images: number;
  duplicate_images: number;
  image_channels: string[];
  min_images_per_class: number;
  max_images_per_class: number;
  average_images_per_class: number;
  class_imbalance_ratio: number;
  resolution_stats: {
    min_width: number;
    min_height: number;
    max_width: number;
    max_height: number;
    avg_width: number;
    avg_height: number;
  };
  class_distribution: Record<string, number>;
  splits: {
    train: number;
    val: number;
    test: number;
  };
}

export interface SampleGalleryItem {
  class_id: number;
  class_name: string;
  image_b64: string;
  dimensions: string;
}

export interface BenchmarkModel {
  model_id: string;
  model_name: string;
  test_accuracy: number;
  macro_f1: number;
  val_accuracy: number;
  latency_ms: number;
  params: string;
  size_mb: number;
  confusion_matrix_plot: string;
  curves_plot: string;
}

export interface EvaluationData {
  benchmark: BenchmarkModel[];
  plots: string[];
  plots_metadata: Record<string, string>;
}

export interface ExperimentRecord {
  exp_id: string;
  model_name: string;
  input_size: string;
  augmentation: string;
  lr: string;
  batch_size: string;
  epochs: string;
  val_acc: string;
  test_acc: string;
  macro_f1: string;
  inference_time_ms: string;
  model_size_mb: string;
  notes: string;
}
