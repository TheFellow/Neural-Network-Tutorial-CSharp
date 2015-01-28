using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml;

namespace NeuralNetwork
{

	#region Transfer functions and their derivatives

	public enum TransferFunction
	{
		None,
		Sigmoid,
		Linear,
		Gaussian,
		RationalSigmoid
	}

	static class TransferFunctions
	{
		public static double Evaluate(TransferFunction tFunc, double input)
		{
			switch (tFunc)
			{
				case TransferFunction.Sigmoid:
					return sigmoid(input);

				case TransferFunction.Linear:
					return linear(input);

				case TransferFunction.Gaussian:
					return gaussian(input);

				case TransferFunction.RationalSigmoid:
					return rationalsigmoid(input);

				case TransferFunction.None:
				default:
					return 0.0;
			}
		}
		public static double EvaluateDerivative(TransferFunction tFunc, double input)
		{
			switch (tFunc)
			{
				case TransferFunction.Sigmoid:
					return sigmoid_derivative(input);

				case TransferFunction.Linear:
					return linear_derivative(input);

				case TransferFunction.Gaussian:
					return gaussian_derivative(input);

				case TransferFunction.RationalSigmoid:
					return rationalsigmoid_derivative(input);

				case TransferFunction.None:
				default:
					return 0.0;
			}
		}

		/* Transfer function definitions */
		private static double sigmoid(double x)
		{
			return 1.0 / (1.0 + Math.Exp(-x));
		}
		private static double sigmoid_derivative(double x)
		{
			return sigmoid(x) * (1 - sigmoid(x));
		}

		private static double linear(double x)
		{
			return x;
		}
		private static double linear_derivative(double x)
		{
			return 1.0;
		}

		private static double gaussian(double x)
		{
			return Math.Exp(-Math.Pow(x, 2));
		}
		private static double gaussian_derivative(double x)
		{
			return -2.0 * x * gaussian(x);
		}

		private static double rationalsigmoid(double x)
		{
			return x / (1.0 + Math.Sqrt(1.0 + x * x));
		}
		private static double rationalsigmoid_derivative(double x)
		{
			double val = Math.Sqrt(1.0 + x * x);
			return 1.0 / (val * (1 + val));
		}

	}

	#endregion

	public class BackPropagationNetwork
	{
		#region Constructors

		public BackPropagationNetwork(int[] layerSizes, TransferFunction[] transferFunctions)
		{
			// Validate the input data
			if (transferFunctions.Length != layerSizes.Length || transferFunctions[0] != TransferFunction.None)
				throw new ArgumentException("Cannot construct a network with these parameters.");

			// Initialize network layers
			layerCount = layerSizes.Length - 1;
			inputSize = layerSizes[0];
			layerSize = new int[layerCount];

			for (int i = 0; i < layerCount; i++)
				layerSize[i] = layerSizes[i + 1];

			transferFunction = new TransferFunction[layerCount];
			for (int i = 0; i < layerCount; i++)
				transferFunction[i] = transferFunctions[i + 1];

			// Start dimensioning arrays
			bias = new double[layerCount][];
			previousBiasDelta = new double[layerCount][];
			delta = new double[layerCount][];
			layerOutput = new double[layerCount][];
			layerInput = new double[layerCount][];

			weight = new double[layerCount][][];
			previousWeightDelta = new double[layerCount][][];

			// Fill 2 dimensional arrays
			for (int l = 0; l < layerCount; l++)
			{
				bias[l] = new double[layerSize[l]];
				previousBiasDelta[l] = new double[layerSize[l]];
				delta[l] = new double[layerSize[l]];
				layerOutput[l] = new double[layerSize[l]];
				layerInput[l] = new double[layerSize[l]];

				weight[l] = new double[l == 0 ? inputSize : layerSize[l - 1]][];
				previousWeightDelta[l] = new double[l == 0 ? inputSize : layerSize[l - 1]][];

				for (int i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]); i++)
				{
					weight[l][i] = new double[layerSize[l]];
					previousWeightDelta[l][i] = new double[layerSize[l]];
				}
			}

			// Initialize the weights
			for (int l = 0; l < layerCount; l++)
			{
				for (int j = 0; j < layerSize[l]; j++)
				{
					bias[l][j] = Gaussian.GetRandomGaussian();
					previousBiasDelta[l][j] = 0.0;
					layerOutput[l][j] = 0.0;
					layerInput[l][j] = 0.0;
					delta[l][j] = 0.0;
				}

				for (int i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]); i++)
				{
					for (int j = 0; j < layerSize[l]; j++)
					{
						weight[l][i][j] = Gaussian.GetRandomGaussian();
						previousWeightDelta[l][i][j] = 0.0;
					}
				}
			}
		}

		public BackPropagationNetwork(string FilePath)
		{
			loaded = false;

			Load(FilePath);

			loaded = true;
		}

		#endregion

		#region Methods

        // Public methods
		public void Run(ref double[] input, out double[] output)
		{
			// Make sure we have enough data
			if (input.Length != inputSize)
				throw new ArgumentException("Input data is not of the correct dimension.");

			// Dimension
			output = new double[layerSize[layerCount - 1]];

			/* Run the network! */
			for (int l = 0; l < layerCount; l++)
			{
				for (int j = 0; j < layerSize[l]; j++)
				{
					double sum = 0.0;
					for (int i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]); i++)
						sum += weight[l][i][j] * (l == 0 ? input[i] : layerOutput[l - 1][i]);

					sum += bias[l][j];
					layerInput[l][j] = sum;

					layerOutput[l][j] = TransferFunctions.Evaluate(transferFunction[l], sum);
				}
			}

			// Copy the output to the output array
			for (int i = 0; i < layerSize[layerCount - 1]; i++)
				output[i] = layerOutput[layerCount - 1][i];
		}

		public double Train(ref double[] input, ref double[] desired, double TrainingRate, double Momentum)
		{
			// Parameter Validation
			if (input.Length != inputSize)
				throw new ArgumentException("Invalid input parameter", "input");
			if (desired.Length != layerSize[layerCount-1])
				throw new ArgumentException("Invalid input parameter", "desired");

			// Local variable
			double error = 0.0, sum = 0.0, weightDelta = 0.0, biasDelta = 0.0;
			double[] output = new double[layerSize[layerCount - 1]];

			// Run the network
			Run(ref input, out output);

			// Back-propagate the error
			for (int l = layerCount - 1; l >= 0; l--)
			{
				// Output layer
				if (l == layerCount - 1)
				{
					for (int k = 0; k < layerSize[l]; k++)
					{
						delta[l][k] = output[k] - desired[k];
						error += Math.Pow(delta[l][k], 2);
						delta[l][k] *= TransferFunctions.EvaluateDerivative(transferFunction[l],
																			layerInput[l][k]);
					}
				}
				else // Hidden layer
				{
					for (int i = 0; i < layerSize[l]; i++)
					{
						sum = 0.0;
						for (int j = 0; j < layerSize[l + 1]; j++)
						{
							sum += weight[l + 1][i][j] * delta[l + 1][j];
						}
						sum *= TransferFunctions.EvaluateDerivative(transferFunction[l], layerInput[l][i]);

						delta[l][i] = sum;
					}
				}
			}

			// Update the weights and biases
			for (int l = 0; l < layerCount; l++)
				for(int i=0; i<(l==0 ? inputSize : layerSize[l-1]); i++)
					for (int j = 0; j < layerSize[l]; j++)
					{
                        weightDelta = TrainingRate * delta[l][j] * (l == 0 ? input[i] : layerOutput[l - 1][i])
                                        + Momentum * previousWeightDelta[l][i][j];
                        weight[l][i][j] -= weightDelta;

                        previousWeightDelta[l][i][j] = weightDelta;
					}

			for(int l=0; l<layerCount; l++)
				for (int i = 0; i < layerSize[l]; i++)
				{
					biasDelta = TrainingRate * delta[l][i];
					bias[l][i] -= biasDelta + Momentum * previousBiasDelta[l][i];

					previousBiasDelta[l][i] = biasDelta;
				}

			return error;
		}

		public void Save(string FilePath)
		{
			if (FilePath == null)
				return;
			XmlWriterSettings settings = new XmlWriterSettings();
			settings.Indent = true;
			settings.IndentChars = "\t";

			XmlWriter writer = XmlWriter.Create(FilePath,settings);
			

			// Begin document
			writer.WriteStartElement("NeuralNetwork");
			writer.WriteAttributeString("Type", "BackPropagation");

			// Parameters element
			writer.WriteStartElement("Parameters");

			writer.WriteElementString("Name", Name);
			writer.WriteElementString("inputSize", inputSize.ToString());
			writer.WriteElementString("layerCount", layerCount.ToString());

			// Layer sizes
			writer.WriteStartElement("Layers");

			for (int l = 0; l < layerCount; l++)
			{
				writer.WriteStartElement("Layer");

				writer.WriteAttributeString("Index", l.ToString());
				writer.WriteAttributeString("Size", layerSize[l].ToString());
				writer.WriteAttributeString("Type", transferFunction[l].ToString());

				writer.WriteEndElement();	// Layer
			}

			writer.WriteEndElement();	// Layers

			writer.WriteEndElement();	// Parameters

			// Weights and biases
			writer.WriteStartElement("Weights");

			for (int l = 0; l < layerCount; l++)
			{
				writer.WriteStartElement("Layer");
				writer.WriteAttributeString("Index", l.ToString());

				for (int j = 0; j < layerSize[l]; j++)
				{
					writer.WriteStartElement("Node");
					writer.WriteAttributeString("Index", j.ToString());
					writer.WriteAttributeString("Bias", bias[l][j].ToString());

					for (int i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]); i++)
					{
						writer.WriteStartElement("Axon");
						writer.WriteAttributeString("Index", i.ToString());

						writer.WriteString(weight[l][i][j].ToString());

						writer.WriteEndElement();	// Axon
					}

					writer.WriteEndElement();	// Node
				}

				writer.WriteEndElement();	// Layer
			}

			writer.WriteEndElement();	// Weights

			writer.WriteEndElement();	// NeuralNetwork

			writer.Flush();
			writer.Close();
		}

		public void Load(string FilePath)
		{
			if (FilePath == null)
				return;

			doc = new XmlDocument();
			doc.Load(FilePath);

			string BasePath = "", NodePath = "";
			double value;

			// Load from xml
			if (xPathValue("NeuralNetwork/@Type") != "BackPropagation")
				return;

			BasePath = "NeuralNetwork/Parameters/";
			Name = xPathValue(BasePath + "Name");

			int.TryParse(xPathValue(BasePath + "inputSize"), out inputSize);
			int.TryParse(xPathValue(BasePath + "layerCount"), out layerCount);

			layerSize = new int[layerCount];
			transferFunction = new TransferFunction[layerCount];

			BasePath = "NeuralNetwork/Parameters/Layers/Layer";
			for (int l = 0; l < layerCount; l++)
			{
				int.TryParse(xPathValue(BasePath + "[@Index='" + l.ToString() + "']/@Size"), out layerSize[l]);
				Enum.TryParse<TransferFunction>(xPathValue(BasePath + "[@Index='" + l.ToString() + "']/@Type"), out transferFunction[l]);
			}

			// Parse the Weights element

			// Start dimensioning arrays
			bias = new double[layerCount][];
			previousBiasDelta = new double[layerCount][];
			delta = new double[layerCount][];
			layerOutput = new double[layerCount][];
			layerInput = new double[layerCount][];

			weight = new double[layerCount][][];
			previousWeightDelta = new double[layerCount][][];

			// Fill 2 dimensional arrays
			for (int l = 0; l < layerCount; l++)
			{
				bias[l] = new double[layerSize[l]];
				previousBiasDelta[l] = new double[layerSize[l]];
				delta[l] = new double[layerSize[l]];
				layerOutput[l] = new double[layerSize[l]];
				layerInput[l] = new double[layerSize[l]];

				weight[l] = new double[l == 0 ? inputSize : layerSize[l - 1]][];
				previousWeightDelta[l] = new double[l == 0 ? inputSize : layerSize[l - 1]][];

				for (int i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]); i++)
				{
					weight[l][i] = new double[layerSize[l]];
					previousWeightDelta[l][i] = new double[layerSize[l]];
				}
			}

			// Initialize the weights
			for (int l = 0; l < layerCount; l++)
			{
				BasePath = "NeuralNetwork/Weights/Layer[@Index='" + l.ToString() + "']/";
				for (int j = 0; j < layerSize[l]; j++)
				{
					NodePath = "Node[@Index='" + j.ToString() + "']/@Bias";
					double.TryParse(xPathValue(BasePath + NodePath), out value);

					bias[l][j] = value;
					previousBiasDelta[l][j] = 0.0;
					layerOutput[l][j] = 0.0;
					layerInput[l][j] = 0.0;
					delta[l][j] = 0.0;
				}

				for (int i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]); i++)
				{
					for (int j = 0; j < layerSize[l]; j++)
					{
						NodePath = "Node[@Index='" + j.ToString() + "']/Axon[@Index='" + i.ToString() + "']";
						double.TryParse(xPathValue(BasePath + NodePath), out value);

						weight[l][i][j] = value;
						previousWeightDelta[l][i][j] = 0.0;
					}
				}
			}

			// "release"
			doc = null;
		}

        public void Nudge(double scalar)
        {
            // Go through all of the weights and biases and augment them
            for (int l = 0; l < layerCount; l++)
            {
                for (int j = 0; j < layerSize[l]; j++)
                {
                    // Nudge the weights
                    for (int i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]); i++)
                    {
                        double w = weight[l][i][j];
                        double u = Gaussian.GetRandomGaussian(0f, w * scalar);
                        weight[l][i][j] += u;
                        previousWeightDelta[l][i][j] = 0f;
                    }

                    // Nudge the bias
                    double b = bias[l][j];
                    double v = Gaussian.GetRandomGaussian(0f, b * scalar);
                    bias[l][j] += v;
                    previousBiasDelta[l][j] = 0f;
                }
            }
        }

        // Private methods
        private string xPathValue(string xPath)
		{
			XmlNode node = doc.SelectSingleNode(xPath);

			if (node == null)
				throw new ArgumentException("Cannot find specified node", xPath);

			return node.InnerText;
		}
		
		#endregion

		#region Public data

		public string Name = "Default";

		#endregion

		#region Private data

		private int layerCount;
		private int inputSize;
		private int[] layerSize;
		private TransferFunction[] transferFunction;

		private double[][] layerOutput;
		private double[][] layerInput;
		private double[][] bias;
		private double[][] delta;
		private double[][] previousBiasDelta;

		private double[][][] weight;
		private double[][][] previousWeightDelta;

		private XmlDocument doc = null;
		private bool loaded = true;

		#endregion
	}

	public static class Gaussian
	{
		private static Random gen = new Random();

		public static double GetRandomGaussian()
		{
			return GetRandomGaussian(0.0, 1.0);
		}

		public static double GetRandomGaussian(double mean, double stddev)
		{
			double rVal1, rVal2;

			GetRandomGaussian(mean, stddev, out rVal1, out rVal2);

			return rVal1;
		}

		public static void GetRandomGaussian(double mean, double stddev, out double val1, out double val2)
		{
			double u, v, s, t;

			do
			{
				u = 2 * gen.NextDouble() - 1;
				v = 2 * gen.NextDouble() - 1;
			} while (u * u + v * v > 1 || (u == 0 && v == 0));

			s = u * u + v * v;
			t = Math.Sqrt((-2.0 * Math.Log(s)) / s);

			val1 = stddev * u * t + mean;
			val2 = stddev * v * t + mean;
		}
	}
}
