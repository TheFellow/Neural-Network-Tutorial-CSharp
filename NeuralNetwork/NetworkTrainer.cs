using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml;

namespace NeuralNetwork
{
    #region Support classes

    public class DataPoint
    {
        public DataPoint() { }
        public DataPoint(double[] Input, double[] Output) { Load(Input, Output); }
        public DataPoint(XmlElement Elem) { Load(Elem); }

        public void Load(double[] Input, double[] Output)
        {
            input = new double[Input.Length]; output = new double[Output.Length];

            Array.Copy(Input, input, Input.Length);
            Array.Copy(Output, output, Output.Length);
        }
        public void Load(XmlElement elem)
        {
            XmlNode nType;
            int lIn, lOut, i;

            nType = elem.SelectSingleNode("Input");
            lIn = nType.ChildNodes.Count;

            input = new double[lIn];
            foreach (XmlNode node in nType.ChildNodes)
            {
                XmlElement Node = (XmlElement)node;
                double val;

                int.TryParse(Node.GetAttribute("Index"), out i);
                double.TryParse(Node.InnerText, out val);

                input[i] = val;
            }

            nType = elem.SelectSingleNode("Output");
            lOut = nType.ChildNodes.Count;

            output = new double[lOut];
            foreach (XmlNode node in nType.ChildNodes)
            {
                XmlElement Node = (XmlElement)node;
                double val;

                int.TryParse(Node.GetAttribute("Index"), out i);
                double.TryParse(Node.InnerText, out val);

                output[i] = val;
            }
        }

        public XmlElement ToXml(XmlDocument doc)
        {
            XmlElement nDataPoint, nType, node;
            int lIn = input.Length, lOut = output.Length;

            nDataPoint = doc.CreateElement("DataPoint");
            nType = doc.CreateElement("Input");

            for (int i = 0; i < lIn; i++)
            {
                node = doc.CreateElement("Data");
                node.SetAttribute("Index", i.ToString());
                node.AppendChild(doc.CreateTextNode(input[i].ToString()));
                nType.AppendChild(node);
            }

            nDataPoint.AppendChild(nType);

            nType = doc.CreateElement("Output");

            for (int i = 0; i < lOut; i++)
            {
                node = doc.CreateElement("Data");
                node.SetAttribute("Index", i.ToString());
                node.AppendChild(doc.CreateTextNode(output[i].ToString()));
                nType.AppendChild(node);
            }

            nDataPoint.AppendChild(nType);

            return nDataPoint;
        }

        public double[] input, output;
        public int inputSize { get { return input.Length; } }
        public int outputSize { get { return output.Length; } }
    }

    public class DataSet
    {
        public DataSet() { Data = new List<DataPoint>(); }
        public XmlElement ToXml(XmlDocument doc)
        {
            XmlElement nDataSet;

            nDataSet = doc.CreateElement("DataSet");

            foreach (DataPoint d in Data)
                nDataSet.AppendChild(d.ToXml(doc));

            return nDataSet;
        }

        public void Load(XmlElement nDataSet)
        {
            foreach (XmlNode node in nDataSet.ChildNodes)
            {
                DataPoint d = new DataPoint((XmlElement)node);
                Data.Add(d);
            }
        }

        public List<DataPoint> Data;
        public int Size
        {
            get { return Data.Count; }
        }
    }

    public class Permutator
    {
        public Permutator(int Size)
        {
            index = new int[Size];
            
            for (int i = 0; i < Size; i++)
                index[i] = i;

            Permute(Size);
        }

        public void Permute(int nTimes)
        {
            int i, j, t;

            for (int n = 0; n < nTimes; n++)
            {
                i = gen.Next(index.Length);
                j = gen.Next(index.Length);

                if (i != j)
                {
                    t = index[i];
                    index[i] = index[j];
                    index[j] = t;
                }
            }
        }

        public int this[int i]
        {
            get
            {
                return index[i];
            }
        }

        private Random gen = new Random();
        private int[] index;
    }

    #endregion

    #region Network training classes

    public class SimpleNetworkTrainer
    {
        // Constructors
        public SimpleNetworkTrainer(BackPropagationNetwork BPN, DataSet DS)
        {
            network = BPN;  dataSet = DS;
            idx = new Permutator(dataSet.Size);
            iterations = 0;

            errorHistory = new List<double>();
        }

        // Training method
        public void TrainDataSet()
        {
            do
            {
                // Prepare to train epoch
                iterations++; error = 0.0;
                idx.Permute(dataSet.Size);

                // Train this epoch
                for (int i = 0; i < dataSet.Size; i++)
                {
                    error += network.Train( ref dataSet.Data[idx[i]].input,
                                            ref dataSet.Data[idx[i]].output,
                                            trainingRate, momentum);
                }

                // Track this error history
                errorHistory.Add(error);

                // Check whether to Nudge
                if (iterations % nudge_window == 0)
                    CheckNudge();

            } while (error > maxError && iterations < maxIterations);
        }

        // Accessor method
        public double[] GetErrorHistory()
        {
            return errorHistory.ToArray();
        }

        // Private method
        private void CheckNudge()
        {
            double oldAvg = 0f, newAvg = 0f;
            int l = errorHistory.Count;

            // Do i enough data?
            if (iterations < 2 * nudge_window) return;

            // Compute our averages and compare
            for (int i = 0; i < nudge_window; i++)
            {
                oldAvg += errorHistory[l - 2 * nudge_window + i];
                newAvg += errorHistory[l - nudge_window + i];
            }

            oldAvg /= nudge_window; newAvg /= nudge_window;

            Console.Write("Iter {0} oldAvg {1:0.0000} newAvg {2:0.0000}", iterations, oldAvg, newAvg);

            if (((double)Math.Abs(newAvg - oldAvg)) / nudge_window < nudge_tolerance)
            {
                network.Nudge(nudge_scale);
                Console.Write(" Nudged.");
            }
            Console.Write("\n");
        }

        // Public fields
        public double maxError = 0.1, maxIterations = 100000;
        public double trainingRate = 0.25, momentum = 0.15;

        public int nudge_window = 50;
        public double nudge_scale = 0.25, nudge_tolerance = 0.0001;

        // Private fields
        private double error;
        private int iterations;
        private Permutator idx;

        private List<double> errorHistory;

        // Training materials
        public BackPropagationNetwork network;
        public DataSet dataSet;
    }

    public class NetworkTrainer
    {
        // Constructor
        public NetworkTrainer(BackPropagationNetwork BPN, DataSet DS)
        {
            network = BPN;
            dataSet = DS;
            Initialize();
        }

        public void Initialize()
        {
            iterations = 0;

            if (idx == null)
                idx = new Permutator(dataSet.Size);
            else
                idx.Permute(dataSet.Size);

            if (errorHistory == null)
                errorHistory = new List<double>();
            else
                errorHistory.Clear();
        }

        // Public train method
        public bool TrainDataSet()
        {
            bool success = true;

            if (success)
                success = _BeforeTrainDataSet();

            if (success)
                success = _TrainDataSetAction();

            if (success)
                success = _AfterTrainDataSet();

            return success;
        }

        // Protected hook methods
        protected virtual bool BeforeTrainDataSet() { return true; }
        protected virtual bool AfterTrainDataSet() { return true; }

        protected virtual bool BeforeTrainEpoch() { return true; }
        protected virtual bool AfterTrainEpoch() { return true; }

        protected virtual bool BeforeTrainDataPoint(ref double[] Input, ref double[] Output, int Index) { return true; }
        protected virtual bool AfterTrainDataPoint(ref double[] Input, ref double[] Output, int Index) { return true; }


        // Private training methods
        private bool _BeforeTrainDataSet()
        {
            Initialize();

            return BeforeTrainDataSet();
        }
        private bool _TrainDataSetAction()
        {
            bool success = true;

            do
            {
                if (success)
                    success = _BeforeTrainEpoch();

                if (success)
                    success = _TrainEpochAction();

                if (success)
                    success = _AfterTrainEpoch();

            } while (error > maxError && iterations < maxIterations && success);

            return success;
        }
        private bool _AfterTrainDataSet()
        {
            return AfterTrainDataSet();
        }

        private bool _BeforeTrainEpoch()
        {
            // Prepare to train epoch
            iterations++;
            error = 0.0;
            idx.Permute(dataSet.Size);

            return BeforeTrainEpoch();
        }
        private bool _TrainEpochAction()
        {
            bool success = true;

            // Train this epoch
            for (int i = 0; i < dataSet.Size && success; i++)
            {
                // Make a local copy of the data point's data
                double[] input = new double[dataSet.Data[idx[i]].inputSize];
                double[] output = new double[dataSet.Data[idx[i]].outputSize];

                Array.Copy(dataSet.Data[idx[i]].input, input, input.Length);
                Array.Copy(dataSet.Data[idx[i]].output, output, output.Length);

                if (success)
                    success = BeforeTrainDataPoint(ref input, ref output, idx[i]);

                if (success)
                    error += network.Train(ref input, ref output, trainingRate, momentum);

                if (success)
                    success = AfterTrainDataPoint(ref input, ref output, idx[i]);
            }

            return success;
        }
        private bool _AfterTrainEpoch()
        {
            // Track this error history
            errorHistory.Add(error);

            // Check whether to Nudge
            if (iterations % nudge_window == 0 && nudge)
                CheckNudge();

            return AfterTrainEpoch();
        }

        // Accessor method
        public double[] GetErrorHistory()
        {
            return errorHistory.ToArray();
        }

        // Private method
        private void CheckNudge()
        {
            double oldAvg = 0f, newAvg = 0f;
            int l = errorHistory.Count;

            // Do i enough data?
            if (iterations < 2 * nudge_window) return;

            // Compute our averages and compare
            for (int i = 0; i < nudge_window; i++)
            {
                oldAvg += errorHistory[l - 2 * nudge_window + i];
                newAvg += errorHistory[l - nudge_window + i];
            }

            oldAvg /= nudge_window; newAvg /= nudge_window;

            //Console.Write("Iter {0} oldAvg {1:0.0000} newAvg {2:0.0000}", iterations, oldAvg, newAvg);

            if (((double)Math.Abs(newAvg - oldAvg)) / nudge_window < nudge_tolerance)
            {
                network.Nudge(nudge_scale);
                //Console.Write(" Nudged.");
            }
            //Console.Write("\n");
        }

        // Public fields
        public double maxError = 0.1, maxIterations = 100000;
        public double trainingRate = 0.25, momentum = 0.15;

        public int nudge_window = 50;
        public bool nudge = true;
        public double nudge_scale = 0.25, nudge_tolerance = 0.0001;

        // Private fields
        private double error;
        private int iterations;
        private Permutator idx;

        private List<double> errorHistory;

        // Training materials
        public BackPropagationNetwork network;
        public DataSet dataSet;
    
    }

    public class BinaryNoiseTrainer : NetworkTrainer
    {

        // Constructor
        public BinaryNoiseTrainer(BackPropagationNetwork BPN, DataSet DS)
                : base(BPN, DS)
        {
            // Additional initialization stuff here
            rnd = new Random();
        }

        // Overrides
        protected override bool  BeforeTrainEpoch()
        {
            // Reset the NoisyData
            NoisyData = new DataSet();

 	        return true;
        }

        protected override bool  BeforeTrainDataPoint(ref double[] Input, ref double[] Output, int Index)
        {
            // Add some noise to the input data

            // Add noise to the input
            for(int i=0; i<Input.Length; i++)
                if(rnd.NextDouble() < _noise_density)
                    Input[i] = (Input[i] == 0.0 ? 1.0 : 0.0);

            // Add this "dirty" data point to the data set
            DataPoint dp = new DataPoint(Input, Output);
            NoisyData.Data.Add(dp);

 	        return true;
        }

        // Private data
        private double _noise_density = 0.10;
        public double noise_density {
            get { return _noise_density;}
            set{
                _noise_density = Math.Min(1.0, Math.Max(0.0, value));
            }
        }

        public DataSet NoisyData;
        private Random rnd;

    }

    #endregion

}
