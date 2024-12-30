public class KPoolsSingleThread
{
    public Random random;
    public int k;
    public int dimensionCount;
    public int poolCount;
    public List<Sample> samples;
    public Dictionary<int, List<Sample>> sampleHistogram;
    public int classCount;
    public int minSamplesPerClass;
    public List<KPool> kPools;

    public KPoolsSingleThread(int dimensionCount, int classSamplesPerPool, int poolCount, List<Sample> samples, Dictionary<int, List<Sample>> sampleHistogram, int minSamplesPerClass, int[] allDimensions)
    {
        this.random = new Random();
        this.dimensionCount = dimensionCount;
        this.samples = samples;
        this.sampleHistogram = sampleHistogram;
        this.classCount = sampleHistogram.Count;
        this.minSamplesPerClass = minSamplesPerClass;

        // create the kPools container
        this.kPools = new List<KPool>();

        // iterate to create pools
        for (int p = 0; p < poolCount; p++)
        {
            // randomly select the required number of dimensions for this pool
            int[] dimensions = allDimensions.OrderBy(x => random.Next()).Take(dimensionCount).ToArray();

            // create an array of indices for this pool
            List<Sample> poolSamples = new List<Sample>(classSamplesPerPool * classCount);

            // iterate over the classes
            foreach (KeyValuePair<int, List<Sample>> sampleHistogramKVP in sampleHistogram)
            {
                // randomly select the required number of samples for this class (without replacement)
                poolSamples.AddRange(sampleHistogramKVP.Value.OrderBy(x => random.Next()).Take(classSamplesPerPool));
            }

            // create a kpool
            kPools.Add(new KPool(dimensions, poolSamples));
        }
    }

    public KPoolsSingleThread(int dimensionCount, int classSamplesPerPool, int poolCount, List<Sample> samples)
    {
        this.random = new Random();
        this.dimensionCount = dimensionCount;
        this.samples = samples;

        // create a histogram of samples by their output class
        this.sampleHistogram = new Dictionary<int, List<Sample>>();
        foreach (Sample sample in samples)
        {
            if (!sampleHistogram.ContainsKey(sample.output))
            {
                sampleHistogram[sample.output] = new List<Sample>();
            }
            sampleHistogram[sample.output].Add(sample);
        }

        // count classes
        this.classCount = sampleHistogram.Count;

        // find the minimum number of samples per class
        this.minSamplesPerClass = int.MaxValue;
        foreach (int key in sampleHistogram.Keys)
        {
            minSamplesPerClass = Math.Min(minSamplesPerClass, sampleHistogram[key].Count);
        }

        // if the class samples per pool is greater than the minimum samples per class, throw
        if (classSamplesPerPool > minSamplesPerClass)
        {
            throw new ArgumentException("classSamplesPerPool is greater than the minimum samples per class");
        }

        // create a list of all available dimensions
        int[] allDimensions = Enumerable.Range(0, samples[0].input.Count).ToArray();

        // create the kPools container
        this.kPools = new List<KPool>();

        // iterate to create pools
        for (int p = 0; p < poolCount; p++)
        {
            // randomly select the required number of dimensions for this pool
            int[] dimensions = allDimensions.OrderBy(x => random.Next()).Take(dimensionCount).ToArray();

            // create an array of indices for this pool
            List<Sample> poolSamples = new List<Sample>(classSamplesPerPool * classCount);

            // iterate over the classes
            foreach (KeyValuePair<int, List<Sample>> sampleHistogramKVP in sampleHistogram)
            {
                // randomly select the required number of samples for this class (without replacement)
                poolSamples.AddRange(sampleHistogramKVP.Value.OrderBy(x => random.Next()).Take(classSamplesPerPool));
            }

            // create a kpool
            kPools.Add(new KPool(dimensions, poolSamples));
        }
    }

    public int Predict(List<float> input)
    {
        // gather votes
        Dictionary<int, int> votes = new Dictionary<int, int>();
        for (int p = 0; p < kPools.Count; p++)
        {
            int vote = kPools[p].Predict(input);
            if (!votes.ContainsKey(vote))
            {
                votes[vote] = 0;
            }
            votes[vote]++;
        }

        // find the majority vote
        int output = -1;
        int maxVotes = -1;
        foreach (int key in votes.Keys)
        {
            if (votes[key] > maxVotes)
            {
                output = key;
                maxVotes = votes[key];
            }
        }

        // return majority vote
        return output;
    }
}