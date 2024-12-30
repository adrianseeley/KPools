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

        Patcher patcher = new Patcher(28, 28, 10);
        List<Sample> trainPatches = patcher.PatchSamples(mnistTrain);

        KPoolsSingleThread patchPools = new KPoolsSingleThread(25, 500, 100, trainPatches);

        int correct = 0;
        int incorrect = 0;
        foreach (Sample testSample in mnistTest)
        {
            List<Sample> testPatches = patcher.PatchSample(testSample);
            Dictionary<int, int> votes = new Dictionary<int, int>();
            foreach (Sample testPatch in testPatches)
            {
                int prediction = patchPools.Predict(testPatch.input);
                if (!votes.ContainsKey(prediction))
                {
                    votes[prediction] = 0;
                }
                votes[prediction]++;
                StringBuilder sb = new StringBuilder();
                sb.Append("Actual: " + testSample.output + " Votes: ");
                foreach (KeyValuePair<int, int> kvp in votes)
                {
                    sb.Append(kvp.Key + ":" + kvp.Value + " ");
                }
                Console.WriteLine(sb.ToString());
            }
            int majorityVote = -1;
            int maxVotes = -1;
            foreach (int key in votes.Keys)
            {
                if (votes[key] > maxVotes)
                {
                    majorityVote = key;
                    maxVotes = votes[key];
                }
            }
            if (majorityVote == testSample.output)
            {
                correct++;
            }
            else
            {
                incorrect++;
            }
            Console.WriteLine("Correct: " + correct + " Incorrect: " + incorrect);
        }
        float fitness = (float)correct / (float)mnistTest.Count;


        /*
                KPoolsSingleThread kPools = new KPoolsSingleThread(28, 50, 50, mnistTrain);
                float fitness = Fitness(kPools, mnistTest);
                */
        Console.WriteLine("Fitness: " + fitness);
        Console.ReadLine();
    }
}

