using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InspectView.Models;
using Microsoft.Win32;
using OpenCvSharp;
using OpenCvSharp.WpfExtensions;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;

namespace InspectView.ViewModels;

public partial class MainViewModel : ObservableObject, IDisposable
{
    private readonly YoloDetector _detector = new();

    [ObservableProperty] private BitmapSource? _displayImage;
    [ObservableProperty] private string _statusText = "Model not loaded";
    [ObservableProperty] private string _modelInfo = "";
    [ObservableProperty] private bool _isProcessing;
    [ObservableProperty] private double _confidenceThreshold = 0.25;
    [ObservableProperty] private int _totalImages;
    [ObservableProperty] private int _passCount;
    [ObservableProperty] private int _failCount;
    [ObservableProperty] private double _avgInferenceMs;
    [ObservableProperty] private string _selectedDomain = "GC10-DET (Steel)";

    public ObservableCollection<DetectionItem> DetectionLog { get; } = [];

    public string[] AvailableDomains { get; } =
        ["GC10-DET (Steel)", "NEU-DET (Steel)", "DeepPCB (PCB)"];

    private readonly Dictionary<string, string> _modelPaths = [];
    private readonly List<double> _inferenceTimes = [];

    public MainViewModel()
    {
        ScanModels();
    }

    private void ScanModels()
    {
        var resultsDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            @"dev\active\yolo26-industrial-vision\results");

        if (!Directory.Exists(resultsDir)) return;

        foreach (var dir in Directory.GetDirectories(resultsDir))
        {
            var onnx = Path.Combine(dir, "weights", "best.onnx");
            if (File.Exists(onnx))
            {
                var name = Path.GetFileName(dir);
                _modelPaths[name] = onnx;
            }
        }
    }

    [RelayCommand]
    private void LoadModel()
    {
        var dialog = new OpenFileDialog
        {
            Title = "Select ONNX Model",
            Filter = "ONNX Models|*.onnx",
            InitialDirectory = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                @"dev\active\yolo26-industrial-vision\results")
        };

        if (dialog.ShowDialog() == true)
        {
            try
            {
                _detector.LoadModel(dialog.FileName, useGpu: true);
                var fileSize = new FileInfo(dialog.FileName).Length / (1024.0 * 1024.0);
                ModelInfo = $"{_detector.ModelName} | {fileSize:F1} MB | GPU";
                StatusText = "Model loaded. Drop images or click 'Open Images'.";
                ResetStats();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Failed to load model:\n{ex.Message}", "Error",
                    MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
    }

    [RelayCommand]
    private async Task LoadQuickModel(string modelKey)
    {
        if (!_modelPaths.TryGetValue(modelKey, out var path))
        {
            MessageBox.Show($"Model not found: {modelKey}");
            return;
        }

        IsProcessing = true;
        try
        {
            await Task.Run(() => _detector.LoadModel(path, useGpu: true));
            var fileSize = new FileInfo(path).Length / (1024.0 * 1024.0);
            ModelInfo = $"{_detector.ModelName} | {fileSize:F1} MB | GPU";
            StatusText = "Model loaded. Drop images or click 'Open Images'.";
            ResetStats();
        }
        catch (Exception ex)
        {
            StatusText = $"Error: {ex.Message}";
        }
        finally
        {
            IsProcessing = false;
        }
    }

    [RelayCommand]
    private async Task OpenImages()
    {
        if (!_detector.IsLoaded)
        {
            MessageBox.Show("Please load a model first.", "Info");
            return;
        }

        var dialog = new OpenFileDialog
        {
            Title = "Select Images",
            Filter = "Images|*.jpg;*.jpeg;*.png;*.bmp",
            Multiselect = true,
            InitialDirectory = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                @"dev\active\yolo26-industrial-vision\datasets\gc10-det\images\val")
        };

        if (dialog.ShowDialog() == true)
        {
            await ProcessImages(dialog.FileNames);
        }
    }

    public async Task ProcessImages(string[] paths)
    {
        if (!_detector.IsLoaded) return;

        IsProcessing = true;
        _detector.ConfidenceThreshold = (float)ConfidenceThreshold;

        foreach (var path in paths)
        {
            try
            {
                var result = await Task.Run(() => _detector.Detect(path));
                _inferenceTimes.Add(result.InferenceMs);

                TotalImages++;
                if (result.Passed) PassCount++;
                else FailCount++;
                AvgInferenceMs = _inferenceTimes.Average();

                // Draw and display
                using var img = Cv2.ImRead(path);
                using var drawn = _detector.DrawDetections(img, result);
                DisplayImage = drawn.ToBitmapSource();

                // Log
                foreach (var det in result.Detections)
                {
                    DetectionLog.Insert(0, new DetectionItem(
                        Path.GetFileName(path),
                        det.ClassName,
                        det.Confidence,
                        result.InferenceMs,
                        result.Passed));
                }

                if (result.Detections.Count == 0)
                {
                    DetectionLog.Insert(0, new DetectionItem(
                        Path.GetFileName(path),
                        "PASS (no defects)",
                        0, result.InferenceMs, true));
                }

                StatusText = $"{Path.GetFileName(path)} | {result.Detections.Count} defects | " +
                             $"{result.InferenceMs:F1}ms | {(result.Passed ? "PASS" : "FAIL")}";
            }
            catch (Exception ex)
            {
                StatusText = $"Error: {Path.GetFileName(path)} - {ex.Message}";
            }
        }

        IsProcessing = false;
    }

    [RelayCommand]
    private void ExportCsv()
    {
        if (DetectionLog.Count == 0)
        {
            MessageBox.Show("No data to export.", "Info");
            return;
        }

        var dialog = new SaveFileDialog
        {
            Title = "Export Results",
            Filter = "CSV|*.csv",
            FileName = $"inspection_{DateTime.Now:yyyyMMdd_HHmmss}.csv"
        };

        if (dialog.ShowDialog() == true)
        {
            var lines = new List<string>
            {
                "File,Defect,Confidence,InferenceMs,Passed"
            };
            lines.AddRange(DetectionLog.Select(d =>
                $"{d.FileName},{d.DefectName},{d.Confidence:F3},{d.InferenceMs:F1},{d.Passed}"));

            File.WriteAllLines(dialog.FileName, lines);
            StatusText = $"Exported {DetectionLog.Count} records to CSV.";
        }
    }

    [RelayCommand]
    private void ResetStats()
    {
        TotalImages = 0;
        PassCount = 0;
        FailCount = 0;
        AvgInferenceMs = 0;
        _inferenceTimes.Clear();
        DetectionLog.Clear();
    }

    partial void OnConfidenceThresholdChanged(double value)
    {
        _detector.ConfidenceThreshold = (float)value;
    }

    public void Dispose()
    {
        _detector.Dispose();
    }
}

public record DetectionItem(
    string FileName,
    string DefectName,
    float Confidence,
    double InferenceMs,
    bool Passed);
