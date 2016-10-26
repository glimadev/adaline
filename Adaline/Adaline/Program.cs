using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Adaline
{
    class Program
    {
        static void Main(string[] args)
        {
            double[,] inputs = {
		        {-1.0, -1.0, -1.0, -1.0 },
				{ 0.1,  0.3,  0.6,  0.5 },
				{ 0.4,  0.7,  0.9,  0.7 },
				{ 0.7,  0.2,  0.8,  0.1 }
            };

            double[] outputs = { 1.0, -1.0, -1.0, 1.0 };

            Console.WriteLine(inputs.Length);

            //Fit data 
            Adaline perceptron = new Adaline(inputs, outputs);
            int epoch = perceptron.Fit();
            Console.WriteLine(epoch);

            double[] data = { -1.0, 0.1, 0.4, 0.7 };
            int r = perceptron.Predict(data);
            Console.WriteLine(r);

            double[] data2 = { -1.0, 0.3, 0.7, 0.2 };
            int r2 = perceptron.Predict(data2);
            Console.WriteLine(r2);

            Console.ReadLine();
        }
    }
}
