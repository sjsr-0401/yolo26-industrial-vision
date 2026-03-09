using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace InspectView.Models;

public record Detection(
    float X1, float Y1, float X2, float Y2,
    float Confidence, int ClassId, string ClassName);

public record InspectionResult(
    List<Detection> Detections,
    double InferenceMs,
    int ImageWidth,
    int ImageHeight,
    string ModelName,
    bool Passed);

public class YoloDetector : IDisposable
{
    private InferenceSession? _session;
    private string _modelName = "";
    private int _imgSize = 1024;
    private float _confThreshold = 0.25f;

    private static readonly string[] ClassNames =
    [
        "crease", "crescent_gap", "inclusion", "oil_spot", "punching_hole",
        "rolled_pit", "silk_spot", "waist_folding", "water_spot", "welding_line"
    ];

    public string ModelName => _modelName;
    public bool IsLoaded => _session != null;
    public float ConfidenceThreshold
    {
        get => _confThreshold;
        set => _confThreshold = Math.Clamp(value, 0.01f, 0.99f);
    }

    public void LoadModel(string onnxPath, bool useGpu = true)
    {
        _session?.Dispose();

        var options = new SessionOptions();
        if (useGpu)
        {
            try { options.AppendExecutionProvider_CUDA(0); }
            catch { /* GPU not available, fall back to CPU */ }
        }

        _session = new InferenceSession(onnxPath, options);
        _modelName = Path.GetFileNameWithoutExtension(
            Path.GetDirectoryName(Path.GetDirectoryName(onnxPath)) ?? "unknown");

        // Detect imgsz from input shape
        var inputShape = _session.InputMetadata.First().Value.Dimensions;
        if (inputShape.Length >= 4 && inputShape[2] > 0)
            _imgSize = inputShape[2];
    }

    public InspectionResult Detect(string imagePath)
    {
        if (_session == null)
            throw new InvalidOperationException("Model not loaded");

        using var img = Cv2.ImRead(imagePath);
        return Detect(img);
    }

    public InspectionResult Detect(Mat img)
    {
        if (_session == null)
            throw new InvalidOperationException("Model not loaded");

        int origH = img.Rows, origW = img.Cols;

        // Preprocess: letterbox resize
        float scale = Math.Min((float)_imgSize / origH, (float)_imgSize / origW);
        int newW = (int)(origW * scale), newH = (int)(origH * scale);
        int dx = (_imgSize - newW) / 2, dy = (_imgSize - newH) / 2;

        using var resized = new Mat();
        Cv2.Resize(img, resized, new Size(newW, newH));

        using var canvas = new Mat(_imgSize, _imgSize, MatType.CV_8UC3, new Scalar(114, 114, 114));
        resized.CopyTo(canvas[new Rect(dx, dy, newW, newH)]);

        // HWC BGR -> CHW RGB normalized
        var tensor = new DenseTensor<float>(new[] { 1, 3, _imgSize, _imgSize });
        unsafe
        {
            byte* ptr = (byte*)canvas.Data;
            for (int y = 0; y < _imgSize; y++)
            {
                for (int x = 0; x < _imgSize; x++)
                {
                    int idx = (y * _imgSize + x) * 3;
                    tensor[0, 2, y, x] = ptr[idx + 0] / 255f; // B -> channel 2
                    tensor[0, 1, y, x] = ptr[idx + 1] / 255f; // G -> channel 1
                    tensor[0, 0, y, x] = ptr[idx + 2] / 255f; // R -> channel 0
                }
            }
        }

        // Inference
        var inputName = _session.InputMetadata.First().Key;
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, tensor)
        };

        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var results = _session.Run(inputs);
        sw.Stop();

        // Parse output: (1, 300, 6) = [x1, y1, x2, y2, conf, class_id]
        var output = results.First().AsTensor<float>();
        var detections = new List<Detection>();

        int numDetections = output.Dimensions[1]; // 300
        for (int i = 0; i < numDetections; i++)
        {
            float conf = output[0, i, 4];
            if (conf < _confThreshold) continue;

            // Map back to original image coordinates
            float x1 = (output[0, i, 0] - dx) / scale;
            float y1 = (output[0, i, 1] - dy) / scale;
            float x2 = (output[0, i, 2] - dx) / scale;
            float y2 = (output[0, i, 3] - dy) / scale;

            // Clamp
            x1 = Math.Clamp(x1, 0, origW);
            y1 = Math.Clamp(y1, 0, origH);
            x2 = Math.Clamp(x2, 0, origW);
            y2 = Math.Clamp(y2, 0, origH);

            int classId = (int)output[0, i, 5];
            string className = classId < ClassNames.Length ? ClassNames[classId] : $"class_{classId}";

            detections.Add(new Detection(x1, y1, x2, y2, conf, classId, className));
        }

        bool passed = detections.Count == 0;

        return new InspectionResult(
            detections, sw.Elapsed.TotalMilliseconds,
            origW, origH, _modelName, passed);
    }

    public Mat DrawDetections(Mat img, InspectionResult result)
    {
        var output = img.Clone();
        var colors = new Scalar[]
        {
            new(0, 0, 255), new(0, 165, 255), new(0, 255, 255),
            new(0, 255, 0), new(255, 0, 0), new(255, 0, 255),
            new(128, 0, 128), new(0, 128, 255), new(255, 255, 0),
            new(128, 128, 0)
        };

        foreach (var det in result.Detections)
        {
            var color = colors[det.ClassId % colors.Length];
            var pt1 = new Point((int)det.X1, (int)det.Y1);
            var pt2 = new Point((int)det.X2, (int)det.Y2);

            Cv2.Rectangle(output, pt1, pt2, color, 2);

            string label = $"{det.ClassName} {det.Confidence:F2}";
            var textSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, 0.5, 1, out int baseline);
            Cv2.Rectangle(output,
                new Point(pt1.X, pt1.Y - textSize.Height - 6),
                new Point(pt1.X + textSize.Width + 4, pt1.Y),
                color, -1);
            Cv2.PutText(output, label,
                new Point(pt1.X + 2, pt1.Y - 4),
                HersheyFonts.HersheySimplex, 0.5, Scalar.White, 1);
        }

        // Status bar
        string status = result.Passed ? "PASS" : $"FAIL ({result.Detections.Count} defects)";
        var statusColor = result.Passed ? new Scalar(0, 200, 0) : new Scalar(0, 0, 220);
        Cv2.Rectangle(output, new Point(0, 0), new Point(output.Cols, 32), statusColor, -1);
        Cv2.PutText(output, $"{status} | {result.InferenceMs:F1}ms | {result.ModelName}",
            new Point(8, 22), HersheyFonts.HersheySimplex, 0.6, Scalar.White, 2);

        return output;
    }

    public void Dispose()
    {
        _session?.Dispose();
        _session = null;
    }
}
