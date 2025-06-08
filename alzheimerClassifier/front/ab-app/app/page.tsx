"use client"

import type React from "react"
import { useState, useCallback, useEffect } from "react"
import { Upload, Download, BarChart3, Zap, Brain, TrendingUp, FileImage, Trash2, Play, Pause } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import Image from "next/image"

interface ClassificationResult {
  id: string
  filename: string
  predictions: { label: string; confidence: number }[]
  processingTime: number
  imageUrl: string
  timestamp: Date
}

interface ModelMetrics {
  accuracy: number
  precision: number
  recall: number
  f1_score: number
  inference_time: number
}


export default function AdvancedImageClassification() {
  const [files, setFiles] = useState<File[]>([])
  const [results, setResults] = useState<ClassificationResult[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingProgress, setProcessingProgress] = useState(0)
  const [currentProcessing, setCurrentProcessing] = useState<string>("")
  const [error, setError] = useState<string | null>(null)
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics>({
    accuracy: 0,
    precision: 0,
    recall: 0,
    f1_score: 0,
    inference_time: 0,
  })

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000"

  const ConfusionMatrixDisplay = () => {
    const [imageSrc, setImageSrc] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
      const fetchConfusionMatrix = async () => {
        try {
          const response = await fetch(`${API_URL}/latest_confusion_matrix`);
          if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
          }
          const blob = await response.blob();
          const objectUrl = URL.createObjectURL(blob);
          setImageSrc(objectUrl);
          setLoading(false);
        } catch (err) {
          setError(err.message);
          setLoading(false);
        }
      };

      fetchConfusionMatrix();

      // Cleanup object URL to prevent memory leaks
      return () => {
        if (imageSrc) URL.revokeObjectURL(imageSrc);
      };
    }, []);

    if (loading) {
      return <div className="text-center p-4">Loading confusion matrix...</div>;
    }

    if (error) {
      return <div className="text-center p-4 text-red-500">Error: {error}</div>;
    }

    return (
      <div className="mt-4">
        <h4 className="text-md font-semibold mb-2">Confusion Matrix</h4>
        {imageSrc && (
          <Image
            src={imageSrc}
            alt="Confusion Matrix"
            width={400}
            height={300}
            className="rounded-lg max-w-full h-auto"
          />
        )}
      </div>
    );
  };

  // Fetch model metrics on component mount
  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await fetch(`${API_URL}/metrics`)
        if (!response.ok) throw new Error("Failed to fetch metrics")
        const data = await response.json()
        setModelMetrics(data)
      } catch (err) {
        console.error("Error fetching metrics:", err)
        setError("Could not load model metrics")
      }
    }
    fetchMetrics()
  }, [API_URL])

  const handleFilesDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const droppedFiles = Array.from(e.dataTransfer.files).filter((file) => file.type.startsWith("image/"))
    setFiles((prev) => [...prev, ...droppedFiles])
  }, [])

  const handleFilesSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files).filter((file) => file.type.startsWith("image/"))
      setFiles((prev) => [...prev, ...selectedFiles])
    }
  }

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const processImages = async () => {
    if (files.length === 0) return

    setIsProcessing(true)
    setProcessingProgress(0)
    setResults([])
    setError(null)

    const newResults: ClassificationResult[] = []

    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      setCurrentProcessing(file.name)

      const formData = new FormData()
      formData.append("file", file)

      try {
        const response = await fetch(`${API_URL}/predict`, {
          method: "POST",
          body: formData,
        })

        if (!response.ok) throw new Error(`Failed to classify ${file.name}`)

        const data = await response.json()
        const result = {
          id: `${Date.now()}-${i}`,
          filename: file.name,
          predictions: data.results[0].predictions,
          processingTime: data.results[0].processingTime,
          imageUrl: URL.createObjectURL(file),
          timestamp: new Date(),
        }
        newResults.push(result)
        setResults([...newResults])
        setProcessingProgress(((i + 1) / files.length) * 100)
      } catch (err) {
        setError(err.message)
        break // Stop processing on error; you could modify to continue if desired
      }
    }

    setIsProcessing(false)
    setCurrentProcessing("")
  }

  const exportResults = () => {
    const csvContent = [
      ["Filename", "Top Prediction", "Confidence", "Processing Time (s)", "Timestamp"],
      ...results.map((result) => [
        result.filename,
        result.predictions[0]?.label || "No prediction",
        result.predictions[0]?.confidence.toFixed(3) || "0",
        result.processingTime.toFixed(3),
        result.timestamp.toISOString(),
      ]),
    ]
      .map((row) => row.join(","))
      .join("\n")

    const blob = new Blob([csvContent], { type: "text/csv" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "classification_results.csv"
    a.click()
  }

  const clearAll = () => {
    setFiles([])
    setResults([])
    setError(null)
  }

  // The rest of the JSX remains unchanged
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto py-8 px-4">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
            Advanced Image Classification System
          </h1>
          <p className="text-lg text-muted-foreground">AI-Powered Image Recognition with Real-time Analytics</p>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Upload Section */}
          <div className="lg:col-span-2 space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Upload className="h-5 w-5" />
                  Image Upload & Processing
                </CardTitle>
                <CardDescription>Upload multiple images for batch classification</CardDescription>
              </CardHeader>
              <CardContent>
                <div
                  className="border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all hover:border-primary/50 hover:bg-primary/5"
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={handleFilesDrop}
                  onClick={() => document.getElementById("file-upload")?.click()}
                >
                  <div className="flex flex-col items-center space-y-4">
                    <div className="rounded-full bg-primary/10 p-4">
                      <FileImage className="h-8 w-8 text-primary" />
                    </div>
                    <div>
                      <p className="text-lg font-medium">Drop images here or click to browse</p>
                      <p className="text-sm text-muted-foreground">Supports JPG, PNG, WebP • Max 10MB per file</p>
                    </div>
                  </div>
                  <input
                    id="file-upload"
                    type="file"
                    accept="image/*"
                    multiple
                    className="hidden"
                    onChange={handleFilesSelect}
                  />
                </div>

                {files.length > 0 && (
                  <div className="mt-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="font-medium">Selected Images ({files.length})</h3>
                      <div className="flex gap-2">
                        <Button onClick={processImages} disabled={isProcessing}>
                          {isProcessing ? (
                            <>
                              <Pause className="h-4 w-4 mr-2" />
                              Processing...
                            </>
                          ) : (
                            <>
                              <Play className="h-4 w-4 mr-2" />
                              Process All
                            </>
                          )}
                        </Button>
                        <Button variant="outline" onClick={clearAll}>
                          <Trash2 className="h-4 w-4 mr-2" />
                          Clear All
                        </Button>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4 max-h-64 overflow-y-auto">
                      {files.map((file, index) => (
                        <div key={index} className="relative group">
                          <div className="aspect-square rounded-lg overflow-hidden border">
                            <Image
                              src={URL.createObjectURL(file) || "/placeholder.svg"}
                              alt={file.name}
                              width={150}
                              height={150}
                              className="object-cover w-full h-full"
                            />
                          </div>
                          <Button
                            variant="destructive"
                            size="sm"
                            className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity"
                            onClick={() => removeFile(index)}
                          >
                            <Trash2 className="h-3 w-3" />
                          </Button>
                          <p className="text-xs text-center mt-1 truncate">{file.name}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {isProcessing && (
                  <div className="mt-6 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Processing: {currentProcessing}</span>
                      <span className="text-sm text-muted-foreground">{Math.round(processingProgress)}%</span>
                    </div>
                    <Progress value={processingProgress} className="w-full" />
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Results Section */}
            {results.length > 0 && (
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2">
                      <BarChart3 className="h-5 w-5" />
                      Classification Results
                    </CardTitle>
                    <Button onClick={exportResults} variant="outline" size="sm">
                      <Download className="h-4 w-4 mr-2" />
                      Export CSV
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4 max-h-96 overflow-y-auto">
                    {results.map((result) => (
                      <div key={result.id} className="border rounded-lg p-4">
                        <div className="flex items-start gap-4">
                          <div className="w-16 h-16 rounded-lg overflow-hidden flex-shrink-0">
                            <Image
                              src={result.imageUrl || "/placeholder.svg"}
                              alt={result.filename}
                              width={64}
                              height={64}
                              className="object-cover w-full h-full"
                            />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center justify-between mb-2">
                              <h4 className="font-medium truncate">{result.filename}</h4>
                              <Badge variant="secondary">{result.processingTime.toFixed(3)}s</Badge>
                            </div>
                            <div className="space-y-2">
                              {result.predictions.slice(0, 3).map((pred, idx) => (
                                <div key={idx} className="flex items-center justify-between">
                                  <span className="text-sm">{pred.label}</span>
                                  <div className="flex items-center gap-2">
                                    <div className="w-24 bg-muted rounded-full h-2">
                                      <div
                                        className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full"
                                        style={{ width: `${pred.confidence * 100}%` }}
                                      />
                                    </div>
                                    <span className="text-xs text-muted-foreground w-12 text-right">
                                      {(pred.confidence * 100).toFixed(1)}%
                                    </span>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Analytics Sidebar */}
          <div className="space-y-6">
            {/* Model Performance */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5" />
                  Model Performance
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Accuracy</span>
                    <span className="font-medium">{modelMetrics.accuracy}%</span>
                  </div>
                  <Progress value={modelMetrics.accuracy} className="h-2" />
                </div>

                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Precision</span>
                    <span className="font-medium">{modelMetrics.precision}%</span>
                  </div>
                  <Progress value={modelMetrics.precision} className="h-2" />
                </div>

                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Recall</span>
                    <span className="font-medium">{modelMetrics.recall}%</span>
                  </div>
                  <Progress value={modelMetrics.recall} className="h-2" />
                </div>

                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">F1-Score</span>
                    <span className="font-medium">{modelMetrics.f1_score}%</span>
                  </div>
                  <Progress value={modelMetrics.f1_score} className="h-2" />
                </div>

                <Separator />

                <div className="flex justify-between items-center">
                  <span className="text-sm">Avg. Inference Time</span>
                  <Badge variant="outline">
                    <Zap className="h-3 w-3 mr-1" />
                    {modelMetrics.inference_time}s
                  </Badge>
                </div>
              </CardContent>
            </Card>

            {/* Processing Statistics */}
            {results.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingUp className="h-5 w-5" />
                    Session Statistics
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600">{results.length}</div>
                      <div className="text-xs text-muted-foreground">Images Processed</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600">
                        {(results.reduce((acc, r) => acc + r.processingTime, 0) / results.length).toFixed(2)}s
                      </div>
                      <div className="text-xs text-muted-foreground">Avg. Time</div>
                    </div>
                  </div>

                  <Separator />

                  <div className="space-y-2">
                    <div className="text-sm font-medium">Top Predictions</div>
                    {Object.entries(
                      results.reduce(
                        (acc, result) => {
                          const topPred = result.predictions[0]?.label
                          if (topPred) {
                            acc[topPred] = (acc[topPred] || 0) + 1
                          }
                          return acc
                        },
                        {} as Record<string, number>,
                      ),
                    )
                      .sort(([, a], [, b]) => b - a)
                      .slice(0, 3)
                      .map(([label, count]) => (
                        <div key={label} className="flex justify-between items-center">
                          <span className="text-sm">{label}</span>
                          <Badge variant="secondary">{count}</Badge>
                        </div>
                      ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Model Architecture Info */}
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Model Architecture</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="text-xs space-y-1">
                  <div className="flex justify-between">
                    <span>Parameters:</span>
                    <span>25.6M</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Layers:</span>
                    <span>50</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Input Size:</span>
                    <span>224×224</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Classes:</span>
                    <span>1,000</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Educational Section */}
        <Card className="mt-8">
          <CardHeader>
            <CardTitle>Deep Learning Pipeline Overview</CardTitle>
            <CardDescription>
              Understanding the complete workflow of image classification model development
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="architecture" className="w-full">
              <TabsList className="grid w-full grid-cols-5">
                <TabsTrigger value="architecture">Architecture</TabsTrigger>
                <TabsTrigger value="training">Training</TabsTrigger>
                <TabsTrigger value="optimization">Optimization</TabsTrigger>
                <TabsTrigger value="deployment">Deployment</TabsTrigger>
                <TabsTrigger value="evaluation">Evaluation</TabsTrigger>
              </TabsList>

              <TabsContent value="architecture" className="mt-6">
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-lg font-semibold mb-3">Neural Network Architecture</h3>
                    <div className="space-y-3">
                      <div className="p-3 bg-blue-50 dark:bg-blue-950 rounded-lg">
                        <div className="font-medium text-sm">Convolutional Layers</div>
                        <div className="text-xs text-muted-foreground">
                          Feature extraction through convolution operations
                        </div>
                      </div>
                      <div className="p-3 bg-green-50 dark:bg-green-950 rounded-lg">
                        <div className="font-medium text-sm">Pooling Layers</div>
                        <div className="text-xs text-muted-foreground">
                          Dimensionality reduction and translation invariance
                        </div>
                      </div>
                      <div className="p-3 bg-purple-50 dark:bg-purple-950 rounded-lg">
                        <div className="font-medium text-sm">Fully Connected Layers</div>
                        <div className="text-xs text-muted-foreground">Classification based on extracted features</div>
                      </div>
                    </div>
                  </div>
                  <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
                    <div className="text-center">
                      <Brain className="h-12 w-12 mx-auto mb-2 text-muted-foreground" />
                      <p className="text-sm text-muted-foreground">CNN Architecture Diagram</p>
                    </div>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="training" className="mt-6">
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-lg font-semibold mb-3">Training Process</h3>
                    <ul className="space-y-2 text-sm">
                      <li className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                        Data augmentation and preprocessing
                      </li>
                      <li className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                        Forward propagation through network
                      </li>
                      <li className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                        Loss calculation using cross-entropy
                      </li>
                      <li className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                        Backpropagation and weight updates
                      </li>
                      <li className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                        Validation and early stopping
                      </li>
                    </ul>
                  </div>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Training Accuracy</span>
                        <span>94.2%</span>
                      </div>
                      <Progress value={94.2} className="h-2" />
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Validation Accuracy</span>
                        <span>92.8%</span>
                      </div>
                      <Progress value={92.8} className="h-2" />
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Training Loss</span>
                        <span>0.23</span>
                      </div>
                      <Progress value={77} className="h-2" />
                    </div>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="optimization" className="mt-6">
                <div className="grid md:grid-cols-3 gap-4">
                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm">Hyperparameter Tuning</CardTitle>
                    </CardHeader>
                    <CardContent className="text-xs space-y-2">
                      <div className="flex justify-between">
                        <span>Learning Rate:</span>
                        <span>0.001</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Batch Size:</span>
                        <span>32</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Optimizer:</span>
                        <span>Adam</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Dropout:</span>
                        <span>0.5</span>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm">Regularization</CardTitle>
                    </CardHeader>
                    <CardContent className="text-xs space-y-2">
                      <div>• L2 Weight Decay</div>
                      <div>• Dropout Layers</div>
                      <div>• Batch Normalization</div>
                      <div>• Data Augmentation</div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm">Transfer Learning</CardTitle>
                    </CardHeader>
                    <CardContent className="text-xs space-y-2">
                      <div>• Pre-trained on ImageNet</div>
                      <div>• Fine-tuned last layers</div>
                      <div>• Frozen early features</div>
                      <div>• Domain adaptation</div>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              <TabsContent value="deployment" className="mt-6">
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-lg font-semibold mb-3">Production Deployment</h3>
                    <div className="space-y-3">
                      <div className="p-3 border rounded-lg">
                        <div className="font-medium text-sm mb-1">Model Optimization</div>
                        <div className="text-xs text-muted-foreground">
                          Quantization, pruning, and knowledge distillation for efficient inference
                        </div>
                      </div>
                      <div className="p-3 border rounded-lg">
                        <div className="font-medium text-sm mb-1">API Integration</div>
                        <div className="text-xs text-muted-foreground">
                          RESTful API with FastAPI/Flask for seamless integration
                        </div>
                      </div>
                      <div className="p-3 border rounded-lg">
                        <div className="font-medium text-sm mb-1">Scalability</div>
                        <div className="text-xs text-muted-foreground">
                          Docker containers with Kubernetes orchestration
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="space-y-4">
                    <div className="text-center p-6 border rounded-lg">
                      <Zap className="h-8 w-8 mx-auto mb-2 text-yellow-500" />
                      <div className="font-medium">Real-time Inference</div>
                      <div className="text-sm text-muted-foreground">45ms average response time</div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="text-center p-4 border rounded-lg">
                        <div className="text-lg font-bold">99.9%</div>
                        <div className="text-xs text-muted-foreground">Uptime</div>
                      </div>
                      <div className="text-center p-4 border rounded-lg">
                        <div className="text-lg font-bold">1000+</div>
                        <div className="text-xs text-muted-foreground">Req/min</div>
                      </div>
                    </div>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="evaluation" className="mt-6">
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-lg font-semibold mb-3">Performance Metrics</h3>
                    <div className="space-y-4">
                      <div>
                        <div className="flex justify-between text-sm mb-2">
                          <span>Top-1 Accuracy</span>
                          <span className="font-medium">94.2%</span>
                        </div>
                        <Progress value={94.2} className="h-2" />
                      </div>
                      <div>
                        <div className="flex justify-between text-sm mb-2">
                          <span>Top-5 Accuracy</span>
                          <span className="font-medium">98.7%</span>
                        </div>
                        <Progress value={98.7} className="h-2" />
                      </div>
                      <div>
                        <div className="flex justify-between text-sm mb-2">
                          <span>Precision (Macro)</span>
                          <span className="font-medium">93.8%</span>
                        </div>
                        <Progress value={93.8} className="h-2" />
                      </div>
                      <div>
                        <div className="flex justify-between text-sm mb-2">
                          <span>Recall (Macro)</span>
                          <span className="font-medium">94.6%</span>
                        </div>
                        <Progress value={94.6} className="h-2" />
                      </div>
                    </div>
                  </div>
                  <div>
                    <div className="aspect-square bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-950 dark:to-purple-950 rounded-lg flex items-center justify-center">
                      <div className="text-center">
                        <BarChart3 className="h-12 w-12 mx-auto mb-2 text-muted-foreground" />
                        <div>{ConfusionMatrixDisplay()}</div>
                      </div>
                    </div>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
