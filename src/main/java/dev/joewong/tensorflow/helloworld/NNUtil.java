package dev.joewong.tensorflow.helloworld;

public class NNUtil {

    public static float[][] create2DFloatArray(int dim1, int dim2) {
        float[][] val = new float[dim1][dim2];
        for (int i=0; i<dim1; i++) {
            for(int j=0; j<dim2; j++) {
                val[i][j] = Ops.getRand();
            }
        }
        return val;
    }

    public static float[][] create2DFloatArray(int dim1, int dim2, float value) {
        float[][] val = new float[dim1][dim2];
        for (int i=0; i<dim1; i++) {
            for(int j=0; j<dim2; j++) {
                val[i][j] = value;
            }
        }
        return val;
    }

    public static float[] createFloatArray(int dim1, float value) {
        float[] val = new float[dim1];
        for (int i=0; i<dim1; i++) {
            val[i] = value;
        }
        return val;
    }

    public static float[] createFloatArray(int dim1) {
        float[] val = new float[dim1];
        for (int i=0; i<dim1; i++) {
            val[i] = Ops.getRand();
        }
        return val;
    }
}
