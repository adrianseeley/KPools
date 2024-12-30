public class Program
{
    public static List<Sample> ReadMNIST(string filename, int max = -1)
    {
        List<Sample> samples = new List<Sample>();
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
            List<float> input = new List<float>();
            for (int i = 1; i < parts.Length; i++)
            {
                input.Add(float.Parse(parts[i]));
            }
            samples.Add(new Sample(input, labelInt));
            if (max != -1 && samples.Count >= max)
            {
                break;
            }
        }
        return samples;
    }

    public static float Fitness(KPools kPools, List<Sample> testSamples)
    {
        int correct = 0;
        foreach (Sample testSample in testSamples)
        {
            int prediction = kPools.Predict(testSample.input);
            if (prediction == testSample.output)
            {
                correct++;
            }
        }
        return (float)correct / (float)testSamples.Count;
    }

    public static void Main()
    {
        Random random = new Random();
        int threadCount = 12;
        int classCount = 10; // there are 10 digits in mnist
        List<Sample> mnistTrain = ReadMNIST("D:/data/mnist_train.csv", max: 1000);
        List<Sample> mnistTest = ReadMNIST("D:/data/mnist_test.csv", max: 1000);


        using TextWriter tw = new StreamWriter("results.csv", false);
        tw.WriteLine("dimensionCount,classSamplesPerPool,poolCount,complexity,fitness");

        List<(int indices, int pools, int dimensions, long complexity)> space = new List<(int indices, int pools, int dimensions, long complexity)>();

        for (int classSamplesPerPool = 1; classSamplesPerPool <= 50; classSamplesPerPool++)
        {
            for (int poolCount = 1; poolCount <= 100; poolCount++)
            {
                for (int dimensionCount = 1; dimensionCount <= mnistTrain[0].input.Count; dimensionCount++)
                {
                    long complexity = dimensionCount * classSamplesPerPool * classCount * poolCount;
                    space.Add((classSamplesPerPool, poolCount, dimensionCount, complexity));
                }
            }
        }

        // sort low to high complexity
        space.Sort((x, y) => x.complexity.CompareTo(y.complexity));

        foreach ((int classSamplesPerPool, int poolCount, int dimensionCount, long complexity) in space)
        {
            KPools kPools = new KPools(dimensionCount, classSamplesPerPool, poolCount, threadCount, mnistTrain);
            float fitness = Fitness(kPools, mnistTest);
            kPools.Release();
            tw.WriteLine($"{dimensionCount},{classSamplesPerPool},{poolCount},{complexity},{fitness}");
            tw.Flush();
            Console.WriteLine($"d: {dimensionCount}, cspp: {classSamplesPerPool}, p: {poolCount}, c: {complexity}, f: {fitness}");
        }
    }
}

