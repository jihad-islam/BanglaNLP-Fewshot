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
    const getColorClasses = () => {
      if (confidence > 0.8)
        return "bg-gradient-to-r from-green-500 to-emerald-600";
      if (confidence > 0.6)
        return "bg-gradient-to-r from-blue-500 to-indigo-600";
      if (confidence > 0.4)
        return "bg-gradient-to-r from-yellow-400 to-orange-500";
      return "bg-gradient-to-r from-slate-400 to-slate-500";
    };

    return (
      <div className="mt-3">
        <div className="flex justify-between text-sm mb-2">
          <span className="text-slate-600 font-medium">Confidence Score</span>
          <span className="font-bold text-slate-900">{percentage}%</span>
        </div>
        <div className="relative">
          <div className="w-full bg-slate-200 rounded-full h-3.5 overflow-hidden shadow-inner">
            <div
              className={`${getColorClasses()} h-3.5 rounded-full transition-all duration-700 ease-out relative shadow-sm`}
              style={{ width: `${percentage}%` }}
            >
              <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderProbabilities = (probabilities) => {
    const sortedProbs = Object.entries(probabilities).sort(
      (a, b) => b[1] - a[1]
    );

    return (
      <div className="mt-4 space-y-3">
        <p className="text-sm font-semibold text-slate-700 flex items-center gap-2">
          <svg
            className="w-4 h-4 text-slate-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            />
          </svg>
          Class Probabilities
        </p>
        <div className="space-y-2">
          {sortedProbs.map(([label, prob], idx) => (
            <div key={label} className="space-y-1">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium text-slate-700">{label}</span>
                <span className="font-bold text-slate-900">
                  {(prob * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-slate-100 rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all duration-500 ${
                    idx === 0
                      ? "bg-gradient-to-r from-blue-500 to-indigo-500"
                      : "bg-slate-300"
                  }`}
                  style={{ width: `${prob * 100}%` }}
                ></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderModelCard = (modelData, title) => {
    if (!modelData) return null;

    return (
      <div className="relative bg-white rounded-xl shadow-xl p-0.5 transition-all duration-300 hover:shadow-2xl group">
        {/* Gradient border effect */}
        <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500 via-purple-500 to-indigo-500 rounded-xl opacity-60 group-hover:opacity-75 transition-opacity blur-[2px]"></div>

        {/* Card content */}
        <div className="relative bg-white rounded-xl p-6">
          <div className="mb-4">
            <h3 className="text-base font-bold text-slate-900">{title}</h3>
          </div>

          <div className="space-y-4">
            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 p-4 rounded-lg border-2 border-blue-200">
              <p className="text-xs text-slate-600 mb-1 font-semibold">
                Predicted Label
              </p>
              <p className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                {modelData.predicted_label}
              </p>
            </div>

            {renderConfidenceBar(modelData.confidence)}
            {renderProbabilities(modelData.probabilities)}

            <div className="pt-3 border-t border-slate-200 flex items-center gap-2">
              <svg
                className="w-3.5 h-3.5 text-slate-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
                />
              </svg>
              <span className="text-xs text-slate-500 font-medium">
                {modelData.model_type}
              </span>
            </div>
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
      return (
        <div className="animate-fadeIn">
          <h2 className="text-xl font-bold text-slate-800 mb-6 flex items-center gap-2">
            <svg
              className="w-6 h-6 text-blue-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
              />
            </svg>
            Model Comparison
          </h2>
          <div className="grid lg:grid-cols-2 gap-6">
            {renderModelCard(result.baseline, "BanglaBERT (Baseline)")}
            {renderModelCard(result.protonet, "Meta Learning (ProtoNet)")}
          </div>
        </div>
      );
    } else if (mode === "baseline" && hasBaseline) {
      return (
        <div className="animate-fadeIn">
          <h2 className="text-xl font-bold text-slate-800 mb-6 flex items-center gap-2">
            <svg
              className="w-6 h-6 text-blue-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
              />
            </svg>
            Analysis Results
          </h2>
          <div className="max-w-2xl mx-auto">
            {renderModelCard(result.baseline, "BanglaBERT (Baseline)")}
          </div>
        </div>
      );
    } else if (mode === "protonet" && hasProtonet) {
      return (
        <div className="animate-fadeIn">
          <h2 className="text-xl font-bold text-slate-800 mb-6 flex items-center gap-2">
            <svg
              className="w-6 h-6 text-blue-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
              />
            </svg>
            Analysis Results
          </h2>
          <div className="max-w-2xl mx-auto">
            {renderModelCard(result.protonet, "Meta Learning (ProtoNet)")}
          </div>
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
      <header className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-700 shadow-xl relative overflow-hidden">
        <div className="absolute inset-0 bg-black/5"></div>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 relative z-10">
          <div className="flex flex-col sm:flex-row items-center gap-4">
            <div className="bg-white/20 backdrop-blur-md p-3 rounded-xl shadow-lg ring-2 ring-white/30 transition-transform hover:scale-105">
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
            <div className="text-center sm:text-left">
              <h1 className="text-3xl font-bold text-white tracking-tight">
                Bangla NLP Research Dashboard
              </h1>
              <p className="text-blue-100 text-sm mt-1">
                Few-Shot & Meta-Learning System for Bangla Text Analysis
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Control Panel */}
        <div className="bg-white rounded-xl shadow-xl p-8 mb-8 border border-slate-200/50">
          <div className="grid sm:grid-cols-2 gap-6 mb-6">
            {/* Task Selection */}
            <div>
              <label className="flex items-center gap-2 text-sm font-semibold text-slate-700 mb-2">
                <svg
                  className="w-4 h-4 text-blue-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
                  />
                </svg>
                Analysis Task
              </label>
              <select
                value={selectedTask}
                onChange={(e) => setSelectedTask(e.target.value)}
                className="w-full px-4 py-3 bg-slate-50 border-[3px] border-slate-300 rounded-lg focus:ring-0 focus:border-blue-500 focus:bg-white transition-all hover:border-slate-400 cursor-pointer text-slate-700 font-medium shadow-sm"
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
              <label className="flex items-center gap-2 text-sm font-semibold text-slate-700 mb-2">
                <svg
                  className="w-4 h-4 text-blue-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
                  />
                </svg>
                Evaluation Mode
              </label>
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value)}
                className="w-full px-4 py-3 bg-slate-50 border-[3px] border-slate-300 rounded-lg focus:ring-0 focus:border-blue-500 focus:bg-white transition-all hover:border-slate-400 cursor-pointer text-slate-700 font-medium shadow-sm"
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
              <label className="flex items-center gap-2 text-sm font-semibold text-slate-700">
                <svg
                  className="w-4 h-4 text-blue-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                  />
                </svg>
                Bangla Text Input
              </label>
              <div className="flex items-center gap-2">
                {inputText && (
                  <span className="text-xs text-slate-500 font-medium">
                    {inputText.length} chars
                  </span>
                )}
                {inputText && (
                  <button
                    onClick={handleClear}
                    className="text-sm text-slate-600 hover:text-red-600 transition-all flex items-center gap-1 px-2.5 py-1 rounded-lg hover:bg-red-50 font-medium"
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
            </div>
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="বাংলা টেক্সট লিখুন... (Enter Bangla text here for analysis)"
              className="w-full px-4 py-3 bg-slate-50 border-[3px] border-slate-300 rounded-lg focus:ring-0 focus:border-blue-500 focus:bg-white transition-all resize-none hover:border-slate-400 shadow-sm text-slate-700 text-base"
              rows="5"
            />
          </div>

          {/* Action Buttons */}
          <div>
            <button
              onClick={handleAnalyze}
              disabled={loading || !inputText.trim()}
              className="w-full bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 hover:from-blue-700 hover:via-indigo-700 hover:to-purple-700 text-white font-bold py-4 px-6 rounded-lg transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2 shadow-lg hover:shadow-xl transform hover:scale-[1.02] disabled:transform-none"
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
            <div className="mt-4 bg-red-50 border-l-4 border-red-500 p-3 rounded-lg shadow-sm animate-shake">
              <div className="flex items-start gap-3">
                <svg
                  className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                    clipRule="evenodd"
                  />
                </svg>
                <div className="flex-1">
                  <h3 className="text-sm font-semibold text-red-800 mb-0.5">
                    Error
                  </h3>
                  <p className="text-sm text-red-700">{error}</p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Results Display */}
        {renderResults()}
      </main>

      {/* Footer */}
      <footer className="mt-12 py-6 text-center border-t border-slate-200 bg-white/50 backdrop-blur-sm">
        <p className="text-slate-600 text-sm font-medium">
          Bangla NLP Research Dashboard • Few-Shot & Meta-Learning System
        </p>
        <p className="text-slate-400 text-xs mt-2">
          Powered by BanglaBERT & Prototypical Networks
        </p>
      </footer>
    </div>
  );
}

export default App;
