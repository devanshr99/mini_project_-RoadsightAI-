
import {
  HealthStatus,
  ModelInfo,
  PredictionResult,
  DatasetSummary,
  SampleGalleryItem,
  EvaluationData,
  ExperimentRecord,
} from '../types';

// Local development uses Vite proxy ('/api' -> http://127.0.0.1:5000).
// Production builds use VITE_API_BASE_URL or the deployed Render backend.
const API_BASE = (
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.DEV
    ? '/api'
    : 'https://mini-project-roadsightai.onrender.com/api')
).replace(/\/+$/, '');

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
    this.name = 'ApiError';
  }
}

// Default timeout: 30s to accommodate server spin-up/cold starts.
// For inference/prediction: 90s to accommodate initial model lazy loading into CPU memory.
const DEFAULT_TIMEOUT_MS = 30_000;
const INFERENCE_TIMEOUT_MS = 90_000;

async function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeoutMs: number = DEFAULT_TIMEOUT_MS
): Promise<Response> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(id);
    return res;
  } catch (err) {
    clearTimeout(id);
    throw err;
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errMessage = `HTTP ${res.status}: ${res.statusText}`;
    try {
      const data = await res.json();
      if (data.error) errMessage = data.error;
    } catch {
      // ignore json parse error
    }
    throw new ApiError(errMessage, res.status);
  }
  return res.json();
}

export const api = {
  async getHealth(): Promise<HealthStatus> {
    const res = await fetchWithTimeout(`${API_BASE}/health`);
    return handleResponse<HealthStatus>(res);
  },

  async getModels(): Promise<ModelInfo[]> {
    const res = await fetchWithTimeout(`${API_BASE}/models`);
    const data = await handleResponse<{ models: ModelInfo[] }>(res);
    return data.models;
  },

  async getClasses(): Promise<Record<string, string>> {
    const res = await fetchWithTimeout(`${API_BASE}/classes`);
    const data = await handleResponse<{ classes: Record<string, string> }>(res);
    return data.classes;
  },

  async predict(formData: FormData): Promise<PredictionResult> {
    const res = await fetchWithTimeout(
      `${API_BASE}/predict`,
      {
        method: 'POST',
        body: formData,
      },
      INFERENCE_TIMEOUT_MS
    );
    return handleResponse<PredictionResult>(res);
  },

  async predictBase64(image_base64: string, model: string): Promise<PredictionResult> {
    const res = await fetchWithTimeout(
      `${API_BASE}/predict`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64, model }),
      },
      INFERENCE_TIMEOUT_MS
    );
    return handleResponse<PredictionResult>(res);
  },

  async getDatasetSummary(): Promise<DatasetSummary> {
    const res = await fetchWithTimeout(`${API_BASE}/dataset/summary`);
    return handleResponse<DatasetSummary>(res);
  },

  async getSample(classId?: number): Promise<{
    class_id: number;
    class_name: string;
    image_b64: string;
    dimensions: string;
    filename: string;
  }> {
    const url = classId !== undefined
      ? `${API_BASE}/dataset/sample?class_id=${classId}`
      : `${API_BASE}/dataset/sample`;

    const res = await fetchWithTimeout(url);
    return handleResponse(res);
  },

  async getSamplesGallery(): Promise<SampleGalleryItem[]> {
    const res = await fetchWithTimeout(`${API_BASE}/dataset/samples-gallery`);
    const data = await handleResponse<{ gallery: SampleGalleryItem[] }>(res);
    return data.gallery;
  },

  async getEvaluation(): Promise<EvaluationData> {
    const res = await fetchWithTimeout(`${API_BASE}/evaluation`);
    return handleResponse<EvaluationData>(res);
  },

  async getExperiments(): Promise<ExperimentRecord[]> {
    const res = await fetchWithTimeout(`${API_BASE}/experiments`);
    const data = await handleResponse<{ experiments: ExperimentRecord[] }>(res);
    return data.experiments;
  },
};
