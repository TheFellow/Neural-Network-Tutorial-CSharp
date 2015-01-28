using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NeuralNetwork;
using System.Xml;
using System.IO;

namespace NeuralNetworkTutorialApp
{
	class Program
	{
		static void Main(string[] args)
		{
            
            // Training Data
            XmlDocument doc = new XmlDocument();
            doc.Load(@"e:\temp\simpleData.xml");

            DataSet ds = new DataSet();
            ds.Load((XmlElement)doc.DocumentElement.ChildNodes[0]);

            // Network to train
            int[] layerSizes = new int[3] { 2, 5, 1 };
            TransferFunction[] tFuncs = new TransferFunction[3] { TransferFunction.None,
                                                                  TransferFunction.Sigmoid,
                                                                  TransferFunction.Linear };

            BackPropagationNetwork bpn = new BackPropagationNetwork(layerSizes, tFuncs);

            // Network trainer!
            NetworkTrainer nt = new NetworkTrainer(bpn, ds);

            nt.maxError = 0.001; nt.maxIterations = 100000;

            nt.nudge_window = 500;

            // Train
            Console.WriteLine("Training...");
            nt.TrainDataSet();
            Console.WriteLine("Done!");

            // Save the network
            nt.network.Save(@"e:\temp\simple.xml");

            // Save the error history
            double[] error = nt.GetErrorHistory();
            string[] filedata = new string[error.Length];
            for (int i = 0; i < error.Length; i++)
                filedata[i] = i.ToString() + " " + error[i].ToString();

            File.WriteAllLines(@"e:\temp\simple_errors.txt", filedata);
            
			// End of program
			Console.WriteLine("\n\nPress Enter...");
			Console.ReadLine();
		}
	}
}
