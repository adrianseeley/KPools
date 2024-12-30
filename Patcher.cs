class Patcher
{
    public int sampleWidth;
    public int sampleHeight;
    public int patchSize;
    public int patchCount;

    public Patcher(int sampleWidth, int sampleHeight, int patchSize)
    {
        this.sampleWidth = sampleWidth;
        this.sampleHeight = sampleHeight;
        this.patchSize = patchSize;
        this.patchCount = (sampleWidth - patchSize) * (sampleHeight - patchSize);
    }

    public List<Sample> PatchSample(Sample sample)
    {
        // create a list of patch samples
        List<Sample> patchSamples = new List<Sample>();

        // iterate through patch start x
        for (int patchStartX = 0; patchStartX < sampleWidth - patchSize; patchStartX++)
        {
            // iterate through patch start y
            for (int patchStartY = 0; patchStartY < sampleHeight - patchSize; patchStartY++)
            {
                // create a new patch sample
                List<float> patchInput = new List<float>(patchSize * patchSize);
                for (int patchWalkX = 0; patchWalkX < patchSize; patchWalkX++)
                {
                    for (int patchWalkY = 0; patchWalkY < patchSize; patchWalkY++)
                    {
                        patchInput.Add(sample.input[(patchStartX + patchWalkX) + ((patchStartY + patchWalkY) * sampleWidth)]);
                    }
                }

                // add the patch sample to the list
                patchSamples.Add(new Sample(patchInput, sample.output));
            }
        }

        // return the list of patch samples
        return patchSamples;
    }

    public List<Sample> PatchSamples(List<Sample> samples)
    {
        // create a list of patch samples
        List<Sample> patchSamples = new List<Sample>();

        // iterate through samples to patch them
        foreach (Sample sample in samples)
        {
            // iterate through patch start x
            for (int patchStartX = 0; patchStartX < sampleWidth - patchSize; patchStartX++)
            {
                // iterate through patch start y
                for (int patchStartY = 0; patchStartY < sampleHeight - patchSize; patchStartY++)
                {
                    // create a new patch sample
                    List<float> patchInput = new List<float>(patchSize * patchSize);
                    for (int patchWalkX = 0; patchWalkX < patchSize; patchWalkX++)
                    {
                        for (int patchWalkY = 0; patchWalkY < patchSize; patchWalkY++)
                        {
                            patchInput.Add(sample.input[(patchStartX + patchWalkX) + ((patchStartY + patchWalkY) * sampleWidth)]);
                        }
                    }

                    // add the patch sample to the list
                    patchSamples.Add(new Sample(patchInput, sample.output));
                }
            }
        }

        // return the list of patch samples
        return patchSamples;
    }
}