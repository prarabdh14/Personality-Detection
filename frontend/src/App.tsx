import React, { useState } from 'react';
import { Upload, Video, Loader2, Home, Activity } from 'lucide-react';

interface PersonalityTraits {
  Openness: number;
  Conscientiousness: number;
  Extraversion: number;
  Agreeableness: number;
  Neuroticism: number;
}

interface PredictionResponse {
  prediction: string;
  traits: PersonalityTraits;
}

function App() {
  const [currentPage, setCurrentPage] = useState<'home' | 'analyze'>('home');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.type.startsWith('video/')) {
        setSelectedFile(file);
        setPreview(URL.createObjectURL(file));
        setPrediction(null);
        setError(null);
      } else {
        setError('Please select a valid video file');
      }
    }
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!selectedFile) {
      setError('Please select a video file');
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('video', selectedFile);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to process video');
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const Navbar = () => (
    <nav className="bg-black/40 backdrop-blur-sm fixed w-full top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-8">
            <button
              onClick={() => setCurrentPage('home')}
              className={`flex items-center space-x-2 ${
                currentPage === 'home'
                  ? 'text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <Home className="w-5 h-5" />
              <span>Home</span>
            </button>
            <button
              onClick={() => setCurrentPage('analyze')}
              className={`flex items-center space-x-2 ${
                currentPage === 'analyze'
                  ? 'text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <Activity className="w-5 h-5" />
              <span>Analyze</span>
            </button>
          </div>
        </div>
      </div>
    </nav>
  );

  const HomePage = () => (
    <div className="min-h-screen pt-16 flex flex-col items-center justify-center text-center px-4">
      <h1 className="text-6xl font-bold mb-6 bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
        Video Analysis AI
      </h1>
      <p className="text-xl text-gray-400 max-w-2xl mb-8">
        Harness the power of artificial intelligence to analyze your videos. Our advanced AI model
        processes your video content and provides detailed insights, making it easier than ever
        to understand and interpret video data.
      </p>
      <div className="space-y-6 max-w-2xl">
        <div className="bg-white/5 p-6 rounded-lg">
          <h2 className="text-2xl font-semibold mb-3">How it works</h2>
          <ol className="text-left text-gray-400 space-y-4">
            <li className="flex items-start">
              <span className="font-bold mr-2">1.</span>
              Upload your video file (supports MP4, AVI, or MOV formats)
            </li>
            <li className="flex items-start">
              <span className="font-bold mr-2">2.</span>
              Our AI model processes the video content
            </li>
            <li className="flex items-start">
              <span className="font-bold mr-2">3.</span>
              Receive detailed analysis and insights
            </li>
          </ol>
        </div>
      </div>
      <button
        onClick={() => setCurrentPage('analyze')}
        className="mt-12 bg-white text-black px-8 py-4 rounded-lg font-semibold hover:bg-gray-200 transition-colors"
      >
        Start Analyzing
      </button>
    </div>
  );

  const AnalyzePage = () => (
    <div className="min-h-screen pt-24 pb-12">
      <div className="container mx-auto px-4">
        <div className="max-w-3xl mx-auto">
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold mb-4">Video Analysis</h1>
            <p className="text-gray-400">Upload your video for AI-powered analysis</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-8">
            <div className="bg-white/5 p-8 rounded-lg border-2 border-dashed border-gray-700 hover:border-white/50 transition-colors">
              <div className="flex flex-col items-center justify-center space-y-4">
                {!preview ? (
                  <>
                    <Upload className="w-12 h-12 text-gray-400" />
                    <div className="text-center">
                      <label htmlFor="video-upload" className="cursor-pointer">
                        <span className="text-white hover:text-gray-300">Click to upload</span>
                        <span className="text-gray-400"> or drag and drop</span>
                        <p className="text-sm text-gray-500">MP4, AVI, or MOV (max. 100MB)</p>
                      </label>
                      <input
                        id="video-upload"
                        type="file"
                        accept="video/*"
                        onChange={handleFileSelect}
                        className="hidden"
                      />
                    </div>
                  </>
                ) : (
                  <div className="w-full">
                    <video
                      src={preview}
                      controls
                      className="w-full rounded-lg"
                    />
                    <button
                      type="button"
                      onClick={() => {
                        setSelectedFile(null);
                        setPreview(null);
                      }}
                      className="mt-4 text-sm text-red-400 hover:text-red-300"
                    >
                      Remove video
                    </button>
                  </div>
                )}
              </div>
            </div>

            {error && (
              <div className="bg-red-900/50 text-red-200 p-4 rounded-lg">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={!selectedFile || isLoading}
              className={`w-full py-3 px-6 rounded-lg flex items-center justify-center space-x-2
                ${!selectedFile || isLoading
                  ? 'bg-gray-800 cursor-not-allowed'
                  : 'bg-white text-black hover:bg-gray-200'
                }`}
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <Video className="w-5 h-5" />
                  <span>Analyze Video</span>
                </>
              )}
            </button>
          </form>

          {prediction && (
            <div className="mt-8 bg-white/5 p-6 rounded-lg">
              <h2 className="text-xl font-semibold mb-4">Analysis Result</h2>
              <p className="text-gray-300 mb-4">{prediction.prediction}</p>
              <div className="space-y-4">
                {Object.entries(prediction.traits).map(([trait, score]) => (
                  <div key={trait} className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">{trait}</span>
                      <span className="text-white">{Math.round(score * 100)}%</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div
                        className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${score * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-black text-white">
      <Navbar />
      {currentPage === 'home' ? <HomePage /> : <AnalyzePage />}
    </div>
  );
}

export default App;