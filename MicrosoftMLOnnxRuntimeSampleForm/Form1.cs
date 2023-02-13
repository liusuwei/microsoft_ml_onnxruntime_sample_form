using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;


namespace WindowsFormsApp23
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            string imageFilePath = "val2017\\000000132622.jpg";
            string modelFilePath = "resnet50-v2-7.onnx";
            // Input shape: float[1, 3, 224, 224]
            // Output shape: float[1, 1000]
            // The Input is normalized to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

            // Read image
            using (var _openFileDialog = new OpenFileDialog())
            {
                _openFileDialog.Filter = "Image files|*.jpg;*jpeg;*bmp;*png|All files (*.*)|*.*";
                if (_openFileDialog.ShowDialog() == DialogResult.OK)
                    imageFilePath = _openFileDialog.FileName;
                else
                    return;
            }
            SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgb24> image = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgb24>(imageFilePath);

            // Resize image
            image.Mutate(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new SixLabors.ImageSharp.Size(224, 224),
                    Mode = ResizeMode.Crop
                });
            });

            // Preprocess image
            Tensor<float> input = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var stddev = new[] { 0.229f, 0.224f, 0.225f };
            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                    for (int x = 0; x < accessor.Width; x++)
                    {
                        input[0, 0, y, x] = ((pixelSpan[x].R / 255f) - mean[0]) / stddev[0];
                        input[0, 1, y, x] = ((pixelSpan[x].G / 255f) - mean[1]) / stddev[1];
                        input[0, 2, y, x] = ((pixelSpan[x].B / 255f) - mean[2]) / stddev[2];
                    }
                }
            });

            // Setup inputs
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("data", input)
            };

            // Run inference
            SessionOptions _options = new SessionOptions();
            _options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
            _options.AppendExecutionProvider_CPU(0);
            var session = new InferenceSession(modelFilePath, _options);
            //var session = new InferenceSession(modelFilePath);
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);

            // Postprocess to get softmax vector
            IEnumerable<float> output = results.First().AsEnumerable<float>();
            float sum = output.Sum(x => (float)Math.Exp(x));
            IEnumerable<float> softmax = output.Select(x => (float)Math.Exp(x) / sum);

            // Extract top 10 predicted classes
            IEnumerable<Prediction> top10 = softmax.Select((x, i) => new Prediction { Label = LabelMap.Labels[i], Confidence = x })
                               .OrderByDescending(x => x.Confidence)
                               .Take(10);

            // Print results to console
            string _str = string.Format("Top 10 predictions for ResNet50 v2...");
            _str += System.Environment.NewLine;
            _str += "--------------------------------------------------------------";
            foreach (var t in top10)
            {
                _str += System.Environment.NewLine;
                _str += $"Label: {t.Label}, Confidence: {t.Confidence}";
            }
            MessageBox.Show(_str);
        }
    }
}
