using System.Text;

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

    public static void Main()
    {
        Random random = new Random();
        List<Sample> mnistTrain = ReadMNIST("D:/data/mnist_train.csv", max: 1000);
        List<Sample> mnistTest = ReadMNIST("D:/data/mnist_test.csv", max: 1000);
        int dimensionsPerPool = 50;
        int classSamplesPerPool = 80;
        int poolCount = 500;

        List<Sample> trainSamples = [
            .. mnistTrain, 
            //.. Augment.Translate(mnistTrain, 28, 28, 1, 0),
            //.. Augment.Translate(mnistTrain, 28, 28, -1, 0),
            //.. Augment.Translate(mnistTrain, 28, 28, 0, 1),
            //.. Augment.Translate(mnistTrain, 28, 28, 0, -1),
        ];
        KPools kPools = new KPools(dimensionsPerPool, classSamplesPerPool, poolCount, trainSamples);
        float testFitness = kPools.Fitness(mnistTest);
        Console.WriteLine("Base Fitness: " + testFitness);

        trainSamples = [
            .. mnistTrain,
            .. Augment.Translate(mnistTrain, 28, 28, 1, 0),
            .. Augment.Translate(mnistTrain, 28, 28, -1, 0),
            .. Augment.Translate(mnistTrain, 28, 28, 0, 1),
            .. Augment.Translate(mnistTrain, 28, 28, 0, -1),
        ];
        kPools = new KPools(dimensionsPerPool, classSamplesPerPool, poolCount, trainSamples);
        testFitness = kPools.Fitness(mnistTest);
        Console.WriteLine("Augment Fitness: " + testFitness);


        Console.ReadLine();
    }
}

