import { useState, useRef, useEffect } from 'react'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const WS_BASE_URL = API_BASE_URL.replace(/^http/, 'ws') // Convert http(s) to ws(s)

// --- Audio Helper Functions ---

const downsampleBuffer = (buffer, sampleRate, outSampleRate) => {
  if (outSampleRate === sampleRate) {
    return buffer;
  }
  if (outSampleRate > sampleRate) {
    throw new Error("Downsampling rate show be smaller than original sample rate");
  }
  const sampleRateRatio = sampleRate / outSampleRate;
  const newLength = Math.round(buffer.length / sampleRateRatio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
    let accum = 0, count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    result[offsetResult] = accum / count;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
};

const floatTo16BitPCM = (output, offset, input) => {
  for (let i = 0; i < input.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, input[i]));
    s = s < 0 ? s * 0x8000 : s * 0x7FFF;
    output.setInt16(offset, s, true);
  }
};

function Dictation() {
  const [isRecording, setIsRecording] = useState(false)
  const [rawTranscript, setRawTranscript] = useState('')
  const [partialTranscript, setPartialTranscript] = useState('') // New: for real-time preview
  const [processedText, setProcessedText] = useState('')
  const [tone, setTone] = useState('neutral')
  const [latency, setLatency] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState(null)
  const [recordingTime, setRecordingTime] = useState(0)
  const [copied, setCopied] = useState(null)
  const [darkMode, setDarkMode] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('darkMode')) || false
    } catch { return false }
  })
  
  const timerRef = useRef(null)
  
  // WebSocket & Audio refs
  const socketRef = useRef(null)
  const audioContextRef = useRef(null)
  const scriptProcessorRef = useRef(null)
  const mediaStreamRef = useRef(null)

  const tones = [
    { value: 'neutral', label: 'Neutral' },
    { value: 'formal', label: 'Formal' },
    { value: 'casual', label: 'Casual' },
    { value: 'concise', label: 'Concise' }
  ]

  useEffect(() => {
    if (isRecording) {
      timerRef.current = setInterval(() => setRecordingTime(prev => prev + 1), 1000)
    } else {
      clearInterval(timerRef.current)
    }
    return () => clearInterval(timerRef.current)
  }, [isRecording])

  useEffect(() => {
    localStorage.setItem('darkMode', JSON.stringify(darkMode))
  }, [darkMode])

  const toggleDarkMode = () => setDarkMode(!darkMode)

  const startRecording = async () => {
    try {
      setError(null)
      setRawTranscript('')
      setPartialTranscript('')
      setProcessedText('')
      setLatency(null)
      setRecordingTime(0)

      // 1. Initialize WebSocket
      socketRef.current = new WebSocket(`${WS_BASE_URL}/ws/transcribe`);
      
      socketRef.current.onopen = async () => {
        console.log("WebSocket connected");
        await startAudioCapture();
      };

      socketRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'final') {
          setRawTranscript(prev => (prev ? prev + " " : "") + data.text);
          setPartialTranscript(''); // Clear partial when final arrives
        } else if (data.type === 'partial') {
          setPartialTranscript(data.text);
        } else if (data.type === 'complete') {
          // Final processed result from pipeline
          setProcessedText(data.processed_text);
          setLatency(data.latency?.total_pipeline_ms || data.latency?.total_latency_ms);
          setIsProcessing(false);
          socketRef.current?.close(); // Clean close
        } else if (data.type === 'error') {
          setError(data.message);
          setIsProcessing(false);
        }
      };

      socketRef.current.onerror = (err) => {
        console.error("WebSocket error", err);
        setError("Connection error. Check backend.");
        stopAudioCapture();
      };

      socketRef.current.onclose = () => {
        console.log("WebSocket closed");
        setIsRecording(false);
      };

    } catch (err) {
      console.error(err);
      setError("Failed to initialize recording");
    }
  };

  const startAudioCapture = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: { 
          channelCount: 1, 
          echoCancellation: true, 
          autoGainControl: true,
          noiseSuppression: true
        } 
      });
      
      mediaStreamRef.current = stream;
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      
      const source = audioContextRef.current.createMediaStreamSource(stream);
      // Buffer size 4096 gives decent latency vs stability
      scriptProcessorRef.current = audioContextRef.current.createScriptProcessor(4096, 1, 1);
      
      source.connect(scriptProcessorRef.current);
      scriptProcessorRef.current.connect(audioContextRef.current.destination);

      scriptProcessorRef.current.onaudioprocess = (e) => {
        if (!isRecording && socketRef.current?.readyState !== WebSocket.OPEN) return;

        const inputData = e.inputBuffer.getChannelData(0);
        // Downsample to 16kHz (Vosk standard)
        const downsampledBuffer = downsampleBuffer(inputData, audioContextRef.current.sampleRate, 16000);
        
        // Convert to 16-bit PCM
        const pcmData = new Int16Array(downsampledBuffer.length);
        const dataView = new DataView(pcmData.buffer);
        floatTo16BitPCM(dataView, 0, downsampledBuffer);
        
        // Send to server
        if (socketRef.current?.readyState === WebSocket.OPEN) {
          socketRef.current.send(pcmData.buffer);
        }
      };

      setIsRecording(true);
    } catch (err) {
      console.error("Audio capture error", err);
      setError("Microphone access denied or error");
      socketRef.current?.close();
    }
  };

  const stopAudioCapture = () => {
    if (scriptProcessorRef.current) {
      scriptProcessorRef.current.disconnect();
      scriptProcessorRef.current.onaudioprocess = null;
      scriptProcessorRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
  };

  const stopRecording = () => {
    stopAudioCapture();
    setIsRecording(false);
    setIsProcessing(true); // UI state: Waiting for final processing

    if (socketRef.current?.readyState === WebSocket.OPEN) {
      // Send EOF signal with tone preference
      socketRef.current.send(JSON.stringify({ type: 'eof', tone: tone }));
    }
  };

  const copyToClipboard = async (text, type) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(type)
      setTimeout(() => setCopied(null), 2000)
    } catch (err) { console.error(err) }
  }

  const clearAll = () => {
    setRawTranscript('')
    setPartialTranscript('')
    setProcessedText('')
    setLatency(null)
    setError(null)
  }

  return (
    <div className={`min-h-screen flex flex-col transition-all duration-500 ease-in-out ${
      darkMode ? 'bg-gradient-to-br from-slate-900 via-gray-900 to-slate-800' : 'bg-gradient-to-br from-gray-100 via-white to-gray-200'
    }`}>
      {/* Header */}
      <header className="p-6 backdrop-blur-md border-b border-white/10 shadow-lg sticky top-0 z-50">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-indigo-500 to-pink-500 bg-clip-text text-transparent flex items-center gap-3">
              <span>🎙️</span> Intelligent Dictation
            </h1>
            <p className={`text-sm mt-1 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              Real-time Streaming Engine (<span className="text-green-500 font-bold">&lt;1500ms</span>)
            </p>
          </div>
          <button onClick={toggleDarkMode} className="p-2 rounded-full text-2xl hover:scale-110 transition-transform">
            {darkMode ? '☀️' : '🌙'}
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 py-8 space-y-6">
        {/* Controls */}
        <div className={`p-6 rounded-2xl shadow-xl border ${darkMode ? 'bg-gray-800/80 border-gray-700' : 'bg-white border-gray-200'}`}>
          <div className="flex flex-wrap gap-6 items-center justify-between">
            <div className="flex items-center gap-4">
              <span className={`font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>Tone:</span>
              <select 
                value={tone} 
                onChange={(e) => setTone(e.target.value)}
                disabled={isRecording}
                className={`p-2 rounded-lg border ${darkMode ? 'bg-gray-700 text-white border-gray-600' : 'bg-white text-gray-800 border-gray-300'}`}
              >
                {tones.map(t => <option key={t.value} value={t.value}>{t.label}</option>)}
              </select>
            </div>

            <div className="flex gap-4">
              {!isRecording ? (
                <button 
                  onClick={startRecording}
                  disabled={isProcessing}
                  className="px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-bold rounded-xl shadow-lg hover:scale-105 transition-all flex items-center gap-2 disabled:opacity-50"
                >
                  <span>🎙️</span> Start Streaming
                </button>
              ) : (
                <button 
                  onClick={stopRecording}
                  className="px-6 py-3 bg-red-500 text-white font-bold rounded-xl shadow-lg hover:scale-105 transition-all flex items-center gap-2 animate-pulse"
                >
                  <span>⏹️</span> Stop ({Math.floor(recordingTime / 60)}:{String(recordingTime % 60).padStart(2, '0')})
                </button>
              )}
              <button 
                onClick={clearAll}
                className={`px-4 py-3 border rounded-xl font-semibold hover:bg-opacity-10 transition-colors ${darkMode ? 'border-gray-600 text-gray-300 hover:bg-white' : 'border-gray-300 text-gray-600 hover:bg-black'}`}
              >
                Clear
              </button>
            </div>
          </div>

          {error && (
            <div className="mt-4 p-4 bg-red-100 border border-red-300 text-red-700 rounded-xl flex items-center gap-2">
              <span>⚠️</span> {error}
            </div>
          )}

          {latency !== null && (
            <div className="mt-4 flex items-center gap-4">
              <div className={`px-4 py-2 rounded-lg font-bold ${latency <= 1500 ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}`}>
                Latency: {latency}ms {latency <= 1500 ? '✓' : '⚠️'}
              </div>
            </div>
          )}
        </div>

        {/* Transcripts Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Raw Panel */}
          <div className={`flex flex-col h-[500px] rounded-2xl shadow-xl overflow-hidden border ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
            <div className="p-4 bg-gradient-to-r from-red-500 to-orange-500 text-white font-bold flex justify-between items-center">
              <span>📝 Live Transcript</span>
              {rawTranscript && <button onClick={() => copyToClipboard(rawTranscript, 'raw')} className="hover:scale-110">{copied === 'raw' ? '✓' : '📋'}</button>}
            </div>
            <div className="flex-1 p-6 overflow-y-auto font-medium leading-relaxed">
              <span className={darkMode ? 'text-gray-300' : 'text-gray-700'}>{rawTranscript}</span>
              <span className="text-gray-400 italic ml-1">{partialTranscript}</span>
              {isRecording && <span className="inline-block w-2 h-4 ml-1 bg-red-500 animate-pulse"/>}
              {!rawTranscript && !partialTranscript && (
                <div className="h-full flex items-center justify-center text-gray-400 italic">
                  Start speaking to see live text...
                </div>
              )}
            </div>
          </div>

          {/* Processed Panel */}
          <div className={`flex flex-col h-[500px] rounded-2xl shadow-xl overflow-hidden border ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
            <div className="p-4 bg-gradient-to-r from-green-500 to-teal-500 text-white font-bold flex justify-between items-center">
              <span>✨ Processed Output</span>
              {processedText && <button onClick={() => copyToClipboard(processedText, 'proc')} className="hover:scale-110">{copied === 'proc' ? '✓' : '📋'}</button>}
            </div>
            <div className="flex-1 p-6 overflow-y-auto font-medium leading-relaxed">
              {isProcessing ? (
                <div className="h-full flex flex-col items-center justify-center text-green-500 gap-2">
                  <div className="w-8 h-8 border-4 border-current border-t-transparent rounded-full animate-spin"></div>
                  <span>Finalizing & Formatting...</span>
                </div>
              ) : (
                <p className={darkMode ? 'text-gray-200' : 'text-gray-800'}>{processedText}</p>
              )}
              {!processedText && !isProcessing && (
                <div className="h-full flex items-center justify-center text-gray-400 italic">
                  Processed text will appear here on stop...
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

export default Dictation