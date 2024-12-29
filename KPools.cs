public class KPools
{
    public Random random;
    public int k;
    public int dimensionCount;
    public int poolCount;
    public int threadCount;
    public Barrier startBarrier;
    public Barrier endBarrier;
    public List<(List<float> input, List<float> output)> samples;
    public List<KPool> kPools;
    public List<float>? predictionInput;
    public int[] predictionIndices;
    public bool dispose;

    public KPools(int dimensionCount, int indiciesPerPool, int poolCount, int threadCount, List<(List<float> input, List<float> output)> samples)
    {
        this.random = new Random();
        this.dimensionCount = dimensionCount;
        this.samples = samples;
        this.kPools = new List<KPool>();
        int[] allDimensions = Enumerable.Range(0, samples[0].input.Count).ToArray();
        for (int p = 0; p < poolCount; p++)
        {
            int[] dimensions = allDimensions.OrderBy(x => random.Next()).Take(dimensionCount).ToArray();
            int[] indices = Enumerable.Range(0, indiciesPerPool).OrderBy(x => random.Next()).Take(indiciesPerPool).ToArray();
            kPools.Add(new KPool(dimensions, indices));
        }
        this.predictionIndices = new int[poolCount];
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

    public List<float> Predict(List<float> input, ref List<float> output)
    {
        predictionInput = input;
        startBarrier.SignalAndWait();
        endBarrier.SignalAndWait();
        for (int i = 0; i < output.Count; i++)
        {
            output[i] = 0;
        }
        for (int p = 0; p < kPools.Count; p++)
        {
            int index = predictionIndices[p];
            for (int i = 0; i < output.Count; i++)
            {
                output[i] += samples[index].output[i];
            }
        }
        for (int i = 0; i < output.Count; i++)
        {
            output[i] /= kPools.Count;
        }
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
                predictionIndices[p] = kPool.Predict(samples, predictionInput);
            }
            endBarrier.SignalAndWait();
        }
    }
}

public class  KPool
{
    public int[] dimensions;
    public int[] indices;

    public KPool(int[] dimensions, int[] indices)
    {
        this.dimensions = dimensions;
        this.indices = indices;
    }

    public int Predict(List<(List<float> input, List<float> output)> samples, List<float> input)
    {
        int closestIndex = -1;
        float closestDistance = float.MaxValue;
        foreach(int index in indices)
        {
            float distance = 0;
            foreach (int dimension in dimensions)
            {
                distance += MathF.Pow(input[dimension] - samples[index].input[dimension], 2);
            }
            distance = MathF.Sqrt(distance);
            if (distance < closestDistance)
            {
                closestDistance = distance;
                closestIndex = index;
            }
        }
        return closestIndex;
    }
}

