package dev.joewong.tensorflow.helloworld;

public class XorDataSet {
    
    public static Float[][] getDataSet() {
        return new Float[][] {
            {0.0F, 0.0F, 0.0F},
            {0.0F, 1.0F, 1.0F},
            {1.0F, 0.0F, 1.0F},
            {1.0F, 1.0F, 0.0F}
        };
    }
}
