public class KPoolsMultiThread
{
    public Random random;
    public int k;
    public int dimensionCount;
    public int poolCount;
    public int threadCount;
    public Barrier startBarrier;
    public Barrier endBarrier;
    public List<Sample> samples;
    public Dictionary<int, List<Sample>> sampleHistogram;
    public int classCount;
    public int minSamplesPerClass;
    public List<KPool> kPools;
    public List<float>? predictionInput;
    public int[] predictions;
    public bool dispose;

    public KPoolsMultiThread(int dimensionCount, int classSamplesPerPool, int poolCount, int threadCount, List<Sample> samples)
    {
        this.random = new Random();
        this.dimensionCount = dimensionCount;
        this.samples = samples;
        
        // create a histogram of samples by their output class
        this.sampleHistogram = new Dictionary<int, List<Sample>>();
        foreach(Sample sample in samples)
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
            foreach(KeyValuePair<int, List<Sample>> sampleHistogramKVP in sampleHistogram)
            {
                // randomly select the required number of samples for this class (without replacement)
                poolSamples.AddRange(sampleHistogramKVP.Value.OrderBy(x => random.Next()).Take(classSamplesPerPool));
            }

            // create a kpool
            kPools.Add(new KPool(dimensions, poolSamples));
        }

        // create thread setup
        this.predictionInput = null;
        this.predictions = new int[poolCount];
        this.threadCount = threadCount; 
        this.startBarrier = new Barrier(threadCount + 1);
        this.endBarrier = new Barrier(threadCount + 1);
        this.dispose = false;
        for (int i = 0; i < threadCount; i++)
        {
            Thread thread = new Thread(PredictThread);
            thread.Start(i);
        }
    }

    public void Release()
    {
        dispose = true;
        startBarrier.SignalAndWait();
    }

    public int Predict(List<float> input)
    {
        // set input and start threads, wait for them to finish
        predictionInput = input;
        startBarrier.SignalAndWait();
        endBarrier.SignalAndWait();

        // gather votes
        Dictionary<int, int> votes = new Dictionary<int, int>();
        for (int p = 0; p < kPools.Count; p++)
        {
            int vote = predictions[p];
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

    private void PredictThread(object? param)
    {
        if (param == null)
        {
            throw new ArgumentNullException(nameof(param));
        }
        int threadIndex = (int)param;
        int poolsPerThread = kPools.Count / threadCount;
        int start = threadIndex * poolsPerThread;
        int end = (threadIndex == threadCount - 1) ? kPools.Count : (threadIndex + 1) * poolsPerThread;

        for (; ;)
        {
            startBarrier.SignalAndWait();
            if (dispose)
            {
                return;
            }
            if (predictionInput == null)
            {
                throw new InvalidOperationException("predictionInput is null");
            }
            for (int p = start; p < end; p++)
            {
                KPool kPool = kPools[p];
                predictions[p] = kPool.Predict(predictionInput);
            }
            endBarrier.SignalAndWait();
        }
    }
}