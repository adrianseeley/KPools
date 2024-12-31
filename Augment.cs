public static class Augment
{
    public static List<Sample> Translate(List<Sample> samples, int width, int height, int dx, int dy)
    {
        List<Sample> augmented = new List<Sample>();
        foreach(Sample sample in samples)
        {
            // transpose into 2d
            float[,] image = new float[width, height];
            int i = 0;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    image[x, y] = sample.input[i++];
                }
            }

            // translate
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int sourceX = x - dx;
                    int sourceY = y - dy;
                    if (sourceX >= 0 && sourceX < width && sourceY >= 0 && sourceY < height)
                    {
                        image[x, y] = image[sourceX, sourceY];
                    }
                    else
                    {
                        image[x, y] = 0;
                    }
                }
            }

            // transpose back into 1d
            List<float> input = new List<float>(width * height);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    input.Add(image[x, y]);
                }
            }

            // add sample
            augmented.Add(new Sample(input, sample.output));
        }
        return augmented;
    }
}

