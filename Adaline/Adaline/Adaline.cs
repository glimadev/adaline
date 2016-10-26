using System;

namespace Adaline
{
    public class Adaline
    {
        private const double _threshold = 0.05;
        private const double _precision = 0.00;
        private double[,] _inputs;
        private double[] _weights;
        private double[] _outputs;
        private int _epoch = 0;

        public Adaline(double[,] inputs, double[] outputs)
        {
            _inputs = inputs;
            _outputs = outputs;
        }

        public int Predict(double[] inputs)
        {
            double u = 0;

            for (int i = 0; i < inputs.Length; i++)
            {
                u += inputs[i] * _weights[i];
            }

            return StepFunction(u);
        }

        public int Fit()
        {
            InitialiseRandomWeights(_inputs);

            double EQMAnterior, EQMAtual;

            do
            {
                EQMAnterior = LeastMeanSquare();

                for (int k = 0; k < _inputs.GetLength(0); k++)
                {
                    double u = 0;

                    for (int i = 0; i < _weights.Length; i++)
                    {
                        u += _inputs[i, k] * _weights[i];
                    }

                    for (int i = 0; i < _weights.Length; i++)
                    {
                        _weights[i] = _weights[i] + _threshold * (_outputs[i] - u) * _inputs[i, k];
                    }
                }

                _epoch++;

                EQMAtual = LeastMeanSquare();

            } while (Math.Abs(EQMAtual - EQMAnterior) > _precision);

            return _epoch;
        }

        private double WeightedSum(int i)
        {
            double u = 0;

            for (int j = 0; j < _inputs.GetLength(1); j++)
            {
                u += _inputs[i, j] * _weights[i];
            }

            return u;
        }

        private void InitialiseRandomWeights(double[,] inputs)
        {
            Random rnd = new Random();

            int i = inputs.GetLength(0);

            _weights = new double[i];

            for (int j = 0; j < i; j++)
            {
                _weights[j] = rnd.NextDouble();
            }
        }

        private int StepFunction(double u)
        {
            return (u >= 0.5) ? 1 : -1;
        }

        private double LeastMeanSquare()
        {
            int i = _outputs.Length;

            double lms = 0d;

            for (int x = 0; x < _inputs.GetLength(0); x++)
            {
                double u = 0;

                for (int y = 0; y < _inputs.GetLength(1); y++)
                {
                    u += _weights[y] * _inputs[y, x];
                }

                lms += Math.Pow(_outputs[x] - u, 2);
            }

            return lms / i;
        }
    }
}
