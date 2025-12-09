import axios from "axios";
import { useEffect, useState } from "react";

const API_BASE_URL = "http://localhost:8000";

function App() {
  const [tasks, setTasks] = useState({});
  const [selectedTask, setSelectedTask] = useState("sentiment");
  const [mode, setMode] = useState("comparison");
  const [inputText, setInputText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [availableModels, setAvailableModels] = useState({
    baseline: false,
    protonet: false,
  });

  // Fetch available tasks on mount
  useEffect(() => {
    fetchTasks();
  }, []);

  // Update available models when task changes
  useEffect(() => {
    if (tasks[selectedTask]) {
      const taskData = tasks[selectedTask];
      setAvailableModels({
        baseline: taskData.has_baseline || false,
        protonet: taskData.has_protonet || false,
      });
    }
  }, [selectedTask, tasks]);

  const fetchTasks = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/tasks`);
      setTasks(response.data);
      // Set first available task as default
      const firstTask = Object.keys(response.data)[0];
      if (firstTask) {
        setSelectedTask(firstTask);
        // Check which models are available for the first task
        const taskData = response.data[firstTask];
        setAvailableModels({
          baseline: taskData.has_baseline || false,
          protonet: taskData.has_protonet || false,
        });
      }
    } catch (err) {
      console.error("Failed to fetch tasks:", err);
    }
  };

  const handleClear = () => {
    setInputText("");
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!inputText.trim()) {
      setError("Please enter some text to analyze");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Map UI mode to API mode
      let apiMode = mode;
      if (mode === "baseline" || mode === "protonet") {
        apiMode = "single";
      }

      const requestData = {
        task: selectedTask,
        text: inputText,
        mode: apiMode,
      };

      // Add model_type for single model requests
      if (mode === "baseline") {
        requestData.model_type = "baseline";
      } else if (mode === "protonet") {
        requestData.model_type = "protonet";
      }

      const response = await axios.post(`${API_BASE_URL}/predict`, requestData);
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to analyze text");
    } finally {
      setLoading(false);
    }
  };

  const renderConfidenceBar = (confidence) => {
    const percentage = (confidence * 100).toFixed(1);
    const color =
      confidence > 0.7
        ? "bg-blue-600"
        : confidence > 0.5
        ? "bg-indigo-500"
        : "bg-slate-500";

    return (
      <div className="mt-2">
        <div className="flex justify-between text-sm mb-1">
          <span className="text-slate-600">Confidence</span>
          <span className="font-semibold text-slate-900">{percentage}%</span>
        </div>
        <div className="w-full bg-slate-200 rounded-full h-3">
          <div
            className={`${color} h-3 rounded-full transition-all duration-500`}
            style={{ width: `${percentage}%` }}
          ></div>
        </div>
      </div>
    );
  };

  const renderProbabilities = (probabilities) => {
    return (
      <div className="mt-3 space-y-2">
        <p className="text-sm font-medium text-slate-700">
          Class Probabilities:
        </p>
        {Object.entries(probabilities).map(([label, prob]) => (
          <div
            key={label}
            className="flex items-center justify-between text-sm"
          >
            <span className="text-slate-600">{label}</span>
            <span className="font-mono text-slate-900">
              {(prob * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    );
  };

  const renderModelCard = (modelData, title, isWinner = false) => {
    if (!modelData) return null;

    return (
      <div
        className={`bg-white rounded-lg shadow-lg p-6 border-2 transition-all ${
          isWinner
            ? "border-green-500 ring-2 ring-green-200 shadow-green-100"
            : "border-slate-200"
        }`}
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold text-slate-900">{title}</h3>
          {isWinner && (
            <span className="px-3 py-1 bg-green-100 text-green-700 text-xs font-semibold rounded-full flex items-center gap-1">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                  clipRule="evenodd"
                />
              </svg>
              Winner - Higher Confidence
            </span>
          )}
        </div>

        <div className="space-y-4">
          <div>
            <p className="text-sm text-slate-600 mb-1">Predicted Label</p>
            <p className="text-2xl font-bold text-blue-600">
              {modelData.predicted_label}
            </p>
          </div>

          {renderConfidenceBar(modelData.confidence)}
          {renderProbabilities(modelData.probabilities)}

          <div className="pt-3 border-t border-slate-200">
            <span className="text-xs text-slate-500">
              Model: {modelData.model_type}
            </span>
          </div>
        </div>
      </div>
    );
  };

  const renderResults = () => {
    if (!result) return null;

    const hasBaseline =
      result.baseline !== null && result.baseline !== undefined;
    const hasProtonet =
      result.protonet !== null && result.protonet !== undefined;

    if (mode === "comparison" && hasBaseline && hasProtonet) {
      const baselineWins =
        result.baseline.confidence > result.protonet.confidence;

      return (
        <div className="mt-8 grid md:grid-cols-2 gap-6">
          {renderModelCard(
            result.baseline,
            "BanglaBERT (Baseline)",
            baselineWins
          )}
          {renderModelCard(
            result.protonet,
            "Meta Learning (ProtoNet)",
            !baselineWins
          )}
        </div>
      );
    } else if (mode === "baseline" && hasBaseline) {
      return (
        <div className="mt-8">
          {renderModelCard(result.baseline, "BanglaBERT (Baseline)")}
        </div>
      );
    } else if (mode === "protonet" && hasProtonet) {
      return (
        <div className="mt-8">
          {renderModelCard(result.protonet, "Meta Learning (ProtoNet)")}
        </div>
      );
    } else if (hasBaseline) {
      return (
        <div className="mt-8">
          {renderModelCard(result.baseline, "BanglaBERT (Baseline)")}
        </div>
      );
    } else if (hasProtonet) {
      return (
        <div className="mt-8">
          {renderModelCard(result.protonet, "Meta Learning (ProtoNet)")}
        </div>
      );
    }

    return null;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-600 to-indigo-700 shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center space-x-4">
            <div className="bg-white/10 backdrop-blur-sm p-3 rounded-lg">
              <svg
                className="w-8 h-8 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                />
              </svg>
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white">
                Bangla NLP Research Dashboard
              </h1>
              <p className="text-blue-100 text-sm mt-1">
                Few-Shot & Meta-Learning System
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Control Panel */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <div className="grid md:grid-cols-2 gap-6 mb-6">
            {/* Task Selection */}
            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-2">
                Analysis Task
              </label>
              <select
                value={selectedTask}
                onChange={(e) => setSelectedTask(e.target.value)}
                className="w-full px-4 py-3 border-2 border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition"
              >
                {Object.keys(tasks).map((task) => (
                  <option key={task} value={task}>
                    {task.charAt(0).toUpperCase() + task.slice(1)} Analysis
                  </option>
                ))}
              </select>
            </div>

            {/* Mode Selection */}
            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-2">
                Evaluation Mode
              </label>
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value)}
                className="w-full px-4 py-3 border-2 border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition"
              >
                {availableModels.baseline && (
                  <option value="baseline">BanglaBERT (Baseline)</option>
                )}
                {availableModels.protonet && (
                  <option value="protonet">Meta Learning (ProtoNet)</option>
                )}
                {availableModels.baseline && availableModels.protonet && (
                  <option value="comparison">Model Comparison</option>
                )}
              </select>
            </div>
          </div>

          {/* Text Input */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-semibold text-slate-700">
                Bangla Text Input
              </label>
              {inputText && (
                <button
                  onClick={handleClear}
                  className="text-sm text-slate-600 hover:text-red-600 transition flex items-center gap-1 px-3 py-1 rounded hover:bg-red-50"
                >
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                  Clear
                </button>
              )}
            </div>
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="বাংলা টেক্সট লিখুন... (Enter Bangla text here)"
              className="w-full px-4 py-3 border-2 border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition resize-none"
              rows="4"
            />
          </div>

          {/* Action Buttons */}
          <div className="grid grid-cols-1 gap-3">
            <button
              onClick={handleAnalyze}
              disabled={loading}
              className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold py-3 px-6 rounded-lg transition duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
            >
              {loading ? (
                <>
                  <svg
                    className="animate-spin h-5 w-5 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <svg
                    className="w-5 h-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M13 10V3L4 14h7v7l9-11h-7z"
                    />
                  </svg>
                  <span>Analyze Text</span>
                </>
              )}
            </button>
          </div>

          {/* Error Display */}
          {error && (
            <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-8 rounded-lg">
              <div className="flex">
                <svg
                  className="h-5 w-5 text-red-400"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                    clipRule="evenodd"
                  />
                </svg>
                <p className="ml-3 text-sm text-red-700">{error}</p>
              </div>
            </div>
          )}
        </div>

        {/* Results Display */}
        {renderResults()}
      </main>

      {/* Footer */}
      <footer className="mt-12 py-6 text-center text-slate-600 text-sm">
        <p>Bangla NLP Research Dashboard • Few-Shot & Meta-Learning System</p>
      </footer>
    </div>
  );
}

export default App;
