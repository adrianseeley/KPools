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

    public static float Fitness(KPoolsSingleThread kPools, List<Sample> testSamples)
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
        List<Sample> mnistTrain = ReadMNIST("D:/data/mnist_train.csv", max: 1000);
        List<Sample> mnistTest = ReadMNIST("D:/data/mnist_test.csv", max: 1000);

        // create a histogram of samples by their output class
        Dictionary<int, List<Sample>> trainSampleHistogram = new Dictionary<int, List<Sample>>();
        foreach (Sample trainSample in mnistTrain)
        {
            if (!trainSampleHistogram.ContainsKey(trainSample.output))
            {
                trainSampleHistogram[trainSample.output] = new List<Sample>();
            }
            trainSampleHistogram[trainSample.output].Add(trainSample);
        }

        // count classes
        int classCount = trainSampleHistogram.Count;

        // find the minimum number of samples per class
        int minSamplesPerClass = int.MaxValue;
        foreach (int key in trainSampleHistogram.Keys)
        {
            minSamplesPerClass = Math.Min(minSamplesPerClass, trainSampleHistogram[key].Count);
        }

        // create a list of all available dimensions
        int[] allDimensions = Enumerable.Range(0, mnistTrain[0].input.Count).ToArray();


        using TextWriter tw = new StreamWriter("results.csv", false);
        tw.WriteLine("dimensionCount,classSamplesPerPool,poolCount,complexity,fitness");

        List<(int classSamplesPerPool, int poolCount, int dimensionCount, long complexity)> space = new List<(int, int, int, long)>();

        for (int classSamplesPerPool = 1; classSamplesPerPool <= minSamplesPerClass; classSamplesPerPool++)
        {
            for (int poolCount = 1; poolCount <= 100; poolCount++)
            {
                for (int dimensionCount = 1; dimensionCount <= allDimensions.Length; dimensionCount++)
                {
                    long complexity = dimensionCount * classSamplesPerPool * classCount * poolCount;
                    space.Add((classSamplesPerPool, poolCount, dimensionCount, complexity));
                }
            }
        }

        // sort low to high complexity
        space.Sort((x, y) => x.complexity.CompareTo(y.complexity));

        Parallel.ForEach(space, new ParallelOptions { MaxDegreeOfParallelism = 10 }, (spaceItem) =>
        {
            KPoolsSingleThread kPools = new KPoolsSingleThread(spaceItem.dimensionCount, spaceItem.classSamplesPerPool, spaceItem.poolCount, mnistTrain, trainSampleHistogram, minSamplesPerClass, allDimensions);
            float fitness = Fitness(kPools, mnistTest);
            tw.WriteLine($"{spaceItem.dimensionCount},{spaceItem.classSamplesPerPool},{spaceItem.poolCount},{spaceItem.complexity},{fitness}");
            tw.Flush();
            Console.WriteLine($"d: {spaceItem.dimensionCount}, cspp: {spaceItem.classSamplesPerPool}, p: {spaceItem.poolCount}, c: {spaceItem.complexity}, f: {fitness}");
        });
    }
}

