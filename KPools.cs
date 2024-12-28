public class KPools
{
    public Random random;
    public int k;
    public int dimensionCount;
    public int poolCount;
    public List<(List<float> input, List<float> output)> samples;
    public List<KPool> kPools;

    public KPools(int k, int dimensionCount, int indiciesPerPool, int poolCount, List<(List<float> input, List<float> output)> samples)
    {
        this.random = new Random();
        this.k = k;
        this.dimensionCount = dimensionCount;
        this.samples = samples;
        this.kPools = new List<KPool>();
        int[] allDimensions = Enumerable.Range(0, samples[0].input.Count).ToArray();
        for (int p = 0; p < poolCount; p++)
        {
            int[] dimensions = allDimensions.OrderBy(x => random.Next()).Take(dimensionCount).ToArray();
            int[] indices = Enumerable.Range(0, indiciesPerPool).OrderBy(x => random.Next()).Take(indiciesPerPool).ToArray();
            kPools.Add(new KPool(k, dimensions, indices));
        }
    }

    public List<float> Predict(List<float> input)
    {
        List<float> output = new List<float>(samples[0].output.Count);
        for (int i = 0; i < samples[0].output.Count; i++)
        {
            output.Add(0);
        }
        foreach (KPool kPool in kPools)
        {
            List<float> kOutput = kPool.Predict(samples, input);
            for (int i = 0; i < samples[0].output.Count; i++)
            {
                output[i] += kOutput[i];
            }
        }
        for (int i = 0; i < samples[0].output.Count; i++)
        {
            output[i] /= kPools.Count;
        }
        return output;
    }
}

public class  KPool
{
    public int k;
    public int[] dimensions;
    public int[] indices;

    public KPool(int k, int[] dimensions, int[] indices)
    {
        this.k = k;
        this.dimensions = dimensions;
        this.indices = indices;
    }

    public List<float> Predict(List<(List<float> input, List<float> output)> samples, List<float> input)
    {
        List<IndexDistance> indexDistances = new List<IndexDistance>();
        foreach(int index in indices)
        {
            float distance = 0;
            foreach (int dimension in dimensions)
            {
                distance += MathF.Pow(input[dimension] - samples[index].input[dimension], 2);
            }
            distance = MathF.Sqrt(distance);
            indexDistances.Add(new IndexDistance(index, distance));
        }
        indexDistances.Sort((a, b) => a.distance.CompareTo(b.distance));
        List<float> output = new List<float>(samples[0].output.Count);
        for (int i = 0; i < samples[0].output.Count; i++)
        {
            output.Add(0);
        }
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < samples[0].output.Count; j++)
            {
                output[j] += samples[indexDistances[i].index].output[j];
            }
        }
        for (int i = 0; i < samples[0].output.Count; i++)
        {
            output[i] /= k;
        }
        return output;
    }
}

public class IndexDistance
{
    public int index;
    public float distance;

    public IndexDistance(int index, float distance)
    {
        this.index = index;
        this.distance = distance;
    }
}

