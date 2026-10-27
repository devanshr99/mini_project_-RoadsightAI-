import React from 'react';
import { 
  Code2, 
  BookOpen, 
  Terminal 
} from 'lucide-react';

export const AboutPage: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto space-y-10 py-4">
      {/* Title */}
      <div className="border-b border-stone-200 pb-5">
        <h1 className="text-2xl font-bold text-stone-900 tracking-tight">
          Project Documentation &amp; Academic Methodology
        </h1>
        <p className="text-xs text-stone-600 mt-1">
          Technical specifications, system architecture, explainability mechanics, and replication guidelines for the B.Tech Computer Science Road Sign Recognition &amp; Explainability System.
        </p>
      </div>

      {/* Project Purpose */}
      <section className="bg-white p-6 rounded-xl border border-stone-200 shadow-xs space-y-3">
        <h2 className="text-sm font-semibold text-stone-900 flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-[#e04a2f]" />
          1. Research Objective &amp; Problem Statement
        </h2>
        <p className="text-xs text-stone-600 leading-relaxed">
          Autonomous navigation and driver assistance systems (ADAS) require real-time, highly accurate recognition of road signs under extreme conditions: low light, motion blur, occlusion, and severe scale variance. Black-box deep networks are vulnerable to adversarial perturbations and spurious correlations.
        </p>
        <p className="text-xs text-stone-600 leading-relaxed">
          <strong>RoadSight AI</strong> addresses both classification robustness and model interpretability by pairing high-accuracy convolutional and residual networks with mathematical <strong>Grad-CAM</strong> visual explanations. This enables engineers to verify that the network is focusing on the traffic sign's actual semantic pictogram rather than background artifacts.
        </p>
      </section>

      {/* System Pipeline Architecture */}
      <section className="bg-white p-6 rounded-xl border border-stone-200 shadow-xs space-y-4">
        <h2 className="text-sm font-semibold text-stone-900 flex items-center gap-2">
          <Code2 className="w-4 h-4 text-stone-800" />
          2. End-to-End Pipeline Architecture
        </h2>

        <div className="p-4 rounded-lg bg-stone-50 border border-stone-200/90 font-mono text-[11px] text-stone-800 space-y-2 overflow-x-auto">
          <div className="font-semibold text-stone-900">Dataflow Sequence:</div>
          <div>[Uploaded RGB Image (Any Size)]</div>
          <div className="text-stone-500 pl-4">&darr; Image verification, RGB conversion, aspect-aware bilinear resize to 64×64</div>
          <div>[Standardized Preprocessing: Tensor [1, 3, 64, 64], Mean/Std Normalization]</div>
          <div className="text-stone-500 pl-4">&darr; Forward pass through instantiated architecture (Strict weights loaded)</div>
          <div>[Logits Vector (43 Classes)] &rarr; Softmax &rarr; Top-3 Class Confidences</div>
          <div className="text-stone-500 pl-4">&darr; Backward pass from target class score to target convolutional layer</div>
          <div>[Grad-CAM Gradients &amp; Activations Pool] &rarr; ReLU &rarr; Jet Colormap &rarr; Blended Overlay</div>
          <div className="text-stone-500 pl-4">&darr; Base64 PNG transmission over REST API (&lt; 30ms total latency)</div>
          <div>[Interactive React Split Slider / Side-by-Side Visualization UI]</div>
        </div>
      </section>

      {/* Mathematical Formulation of Grad-CAM */}
      <section className="bg-white p-6 rounded-xl border border-stone-200 shadow-xs space-y-3">
        <h2 className="text-sm font-semibold text-stone-900">
          3. Mathematical Formulation of Grad-CAM
        </h2>
        <p className="text-xs text-stone-600 leading-relaxed">
          Grad-CAM calculates the gradient of the unnormalized score <code className="bg-stone-100 px-1 py-0.5 rounded font-mono">y^c</code> for class <code className="bg-stone-100 px-1 py-0.5 rounded font-mono">c</code> with respect to feature activation maps <code className="bg-stone-100 px-1 py-0.5 rounded font-mono">A^k</code> of the target convolutional layer:
        </p>

        <div className="p-3 bg-stone-100/70 rounded-lg text-center font-mono text-xs text-stone-900 border border-stone-200">
          &alpha;_k^c = (1 / Z) &sum;_i &sum;_j (&part;y^c / &part;A_ij^k)
        </div>

        <p className="text-xs text-stone-600 leading-relaxed">
          The neuron importance weights <code className="bg-stone-100 px-1 py-0.5 rounded font-mono">&alpha;_k^c</code> represent the importance of feature map <code className="bg-stone-100 px-1 py-0.5 rounded font-mono">k</code> for class <code className="bg-stone-100 px-1 py-0.5 rounded font-mono">c</code>. A weighted combination of forward activation maps followed by a rectified linear unit (ReLU) generates the localized saliency map:
        </p>

        <div className="p-3 bg-stone-100/70 rounded-lg text-center font-mono text-xs text-stone-900 border border-stone-200">
          L_Grad-CAM^c = ReLU( &sum;_k &alpha;_k^c A^k )
        </div>
      </section>

      {/* Local-First & Reproducibility */}
      <section className="bg-white p-6 rounded-xl border border-stone-200 shadow-xs space-y-4">
        <h2 className="text-sm font-semibold text-stone-900 flex items-center gap-2">
          <Terminal className="w-4 h-4 text-emerald-700" />
          4. Local Execution &amp; Startup Commands
        </h2>

        <div className="space-y-3 text-xs">
          <div>
            <span className="font-semibold text-stone-800 block mb-1">A. Start Backend REST API:</span>
            <pre className="p-3 rounded-lg bg-stone-900 text-stone-100 font-mono text-[11px] overflow-x-auto">
python app.py
# Runs Flask server on http://127.0.0.1:5000
            </pre>
          </div>

          <div>
            <span className="font-semibold text-stone-800 block mb-1">B. Start Frontend Application:</span>
            <pre className="p-3 rounded-lg bg-stone-900 text-stone-100 font-mono text-[11px] overflow-x-auto">
cd frontend
npm run dev
# Serves React/Vite interface on http://localhost:5173
            </pre>
          </div>

          <div>
            <span className="font-semibold text-stone-800 block mb-1">C. Run Automated Pytest Suite:</span>
            <pre className="p-3 rounded-lg bg-stone-900 text-stone-100 font-mono text-[11px] overflow-x-auto">
pytest tests/
# Executes 29 automated tests verifying models, shape assertions, Grad-CAM hooks, and REST endpoints
            </pre>
          </div>
        </div>
      </section>

      {/* References */}
      <section className="p-5 rounded-xl border border-stone-200 bg-stone-50 text-xs text-stone-600 space-y-2">
        <h3 className="font-semibold text-stone-800">Academic Citations &amp; References:</h3>
        <ol className="list-decimal list-inside space-y-1 font-mono text-[11px]">
          <li>Stallkamp et al., "Man vs. Computer: Benchmarking machine learning algorithms for traffic sign recognition," Neural Networks, 2012.</li>
          <li>Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," ICCV, 2017.</li>
          <li>Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks," CVPR, 2018.</li>
          <li>He et al., "Deep Residual Learning for Image Recognition," CVPR, 2016.</li>
        </ol>
      </section>
    </div>
  );
};
