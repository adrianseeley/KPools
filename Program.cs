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
        int correct = 0;
        int incorrect = 0;
        List<float> predictOutput = new List<float>(test[0].output.Count);
        for (int i = 0; i < test[0].output.Count; i++)
        {
            predictOutput.Add(0);
        }
        for (int testIndex = 0; testIndex < test.Count; testIndex++)
        {
            (List<float> input, List<float> output) sample = test[testIndex];
            kPools.Predict(sample.input, ref predictOutput);
            int predictLabel = predictOutput.IndexOf(predictOutput.Max());
            int actualLabel = sample.output.IndexOf(sample.output.Max());
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
        if (verbose)
        {
            Console.WriteLine();
        }
        return ((float)correct) / ((float)test.Count);
    }

    public static void Main()
    {
        Random random = new Random();
        int threadCount = 12;
        List<(List<float> input, List<float> output)> mnistTrain = ReadMNIST("D:/data/mnist_train.csv", max: 1000);
        List<(List<float> input, List<float> output)> mnistTest = ReadMNIST("D:/data/mnist_test.csv", max: 1000);

        using TextWriter tw = new StreamWriter("results.csv", false);
        tw.WriteLine("dimensionCount,indicesPerPool,poolCount,fitness");

        /*for (int indicesPerPool = 1; indicesPerPool <= 500; indicesPerPool++)
        {
            for (int poolCount = 1; poolCount <= 1000; poolCount += 1)
            {
                for (int dimensionCount = 1; dimensionCount <= mnistTrain[0].input.Count; dimensionCount++)
                {*/
        for (; ; )
        {

            int dimensionCount = random.Next(1, mnistTest[0].input.Count);
            int indicesPerPool = random.Next(1, 500);
            int poolCount = random.Next(1, 1000);
            KPools kPools = new KPools(dimensionCount: dimensionCount, indiciesPerPool: indicesPerPool, poolCount: poolCount, threadCount: threadCount, samples: mnistTrain);
            float fitness = OneHotFitness(kPools, mnistTest, verbose: true);
            kPools.Release();
            tw.WriteLine($"{dimensionCount},{indicesPerPool},{poolCount},{fitness}");
            tw.Flush();
            Console.WriteLine($"d: {dimensionCount}, i: {indicesPerPool}, p: {poolCount}, f: {fitness}");
        }
                /*}
            }
        }*/
    }
}

