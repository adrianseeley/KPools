public class Program
{
    public static List<(List<float> input, List<float> output)> ReadMNIST(string filename, int max = -1)
    {
        List<(List<float> input, List<float> output)> data = new List<(List<float> input, List<float> output)>();
        string[] lines = File.ReadAllLines(filename);
        for (int lineIndex = 1; lineIndex < lines.Length; lineIndex++) // skip headers
        {
            string line = lines[lineIndex].Trim();
            if (line.Length == 0)
            {
                continue; // skip empty lines
            }
            string[] parts = line.Split(',');
            int labelInt = int.Parse(parts[0]);
            List<float> labelOneHot = new List<float>();
            for (int i = 0; i <= 9; i++)
            {
                if (i == labelInt)
                {
                    labelOneHot.Add(1);
                }
                else
                {
                    labelOneHot.Add(0);
                }
            }
            List<float> input = new List<float>();
            for (int i = 1; i < parts.Length; i++)
            {
                input.Add(float.Parse(parts[i]));
            }
            data.Add((input, labelOneHot));
            if (max != -1 && data.Count >= max)
            {
                break;
            }
        }
        return data;
    }

    public static float OneHotFitness(KPools kPools, List<(List<float> input, List<float> output)> test, bool verbose)
    {
        object trackLock = new object();
        int correct = 0;
        int incorrect = 0;
        Parallel.For(0, test.Count, (index) =>
        {
            int localIndex = index;
            (List<float> input, List<float> output) sample = test[localIndex];
            List<float> predictOutput = kPools.Predict(sample.input);
            int predictLabel = predictOutput.IndexOf(predictOutput.Max());
            int actualLabel = sample.output.IndexOf(sample.output.Max());
            lock (trackLock)
            {
                if (predictLabel == actualLabel)
                {
                    correct++;
                }
                else
                {
                    incorrect++;
                }
                if (verbose)
                {
                    Console.Write($"\rCorrect: {correct}, Incorrect: {incorrect}, Fitness: {((float)correct / (correct + incorrect))}");
                }
            }
        });
        if (verbose)
        {
            Console.WriteLine();
        }
        return ((float)correct) / ((float)test.Count);
    }

    public static void Main()
    {
        List<(List<float> input, List<float> output)> mnistTrain = ReadMNIST("D:/data/mnist_train.csv", max: 1000);
        List<(List<float> input, List<float> output)> mnistTest = ReadMNIST("D:/data/mnist_test.csv");

        using TextWriter tw = new StreamWriter("results.csv");
        tw.WriteLine("k,dimensionCount,indicesPerPool,poolCount,fitness");
        for (int k = 1; k <= 10; k++)
        {
            for (int dimensionCount = 1; dimensionCount <= 30; dimensionCount++)
            {
                for (int indicesPerPool = k * 2; indicesPerPool <= 500; indicesPerPool += 5)
                {
                    for (int poolCount = 1; poolCount <= 1000; poolCount += 5)
                    {
                        KPools kPools = new KPools(k: k, dimensionCount: dimensionCount, indiciesPerPool: indicesPerPool, poolCount: poolCount, samples: mnistTrain);
                        float fitness = OneHotFitness(kPools, mnistTest, verbose: false);
                        tw.WriteLine($"{k},{dimensionCount},{indicesPerPool},{poolCount},{fitness}");
                        tw.Flush();
                    }
                }
            }
        }
    }
}

