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
        int incorrect = 0;
        foreach (Sample testSample in testSamples)
        {
            int prediction = kPools.Predict(testSample.input);
            if (prediction == testSample.output)
            {
                correct++;
            }
            else
            {
                incorrect++;
            }
            Console.WriteLine("Correct: " + correct + " Incorrect: " + incorrect);
        }
        return (float)correct / (float)testSamples.Count;
    }

    public static void Main()
    {
        Random random = new Random();
        List<Sample> mnistTrain = ReadMNIST("./mnist_train.csv", max: 1000);
        List<Sample> mnistTest = ReadMNIST("./mnist_test.csv", max: 1000);

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

        KPoolsSingleThread kPools = new KPoolsSingleThread(28, minSamplesPerClass, 50, mnistTrain, trainSampleHistogram, minSamplesPerClass, allDimensions);
        float fitness = Fitness(kPools, mnistTest);
        Console.WriteLine("Fitness: " + fitness);
        Console.ReadLine();
    }
}

