public class KPool
{
    public int[] dimensions;
    public List<Sample> poolSamples;

    public KPool(int[] dimensions, List<Sample> poolSamples)
    {
        this.dimensions = dimensions;
        this.poolSamples = poolSamples;
    }

    public int Predict(List<float> input)
    {
        int closestClass = -1;
        float closestDistance = float.MaxValue;
        foreach (Sample poolSample in poolSamples)
        {
            float distance = 0;
            foreach (int dimension in dimensions)
            {
                distance += Math.Abs(input[dimension] - poolSample.input[dimension]);
            }
            if (distance < closestDistance)
            {
                closestDistance = distance;
                closestClass = poolSample.output;
            }
        }
        return closestClass;
    }
}