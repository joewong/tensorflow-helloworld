package dev.joewong.tensorflow.helloworld;

public class DisplayFloatValues {

    public String displayString(float value) {
        return " => " + value;
    }

    public String displayString(float[] item) {
        String line = "";
        for (int j=0; j<item.length; j++) {
            line += displayString(item[j]);
            if (j < item.length-1) {
                line += "\n";
            }
        }
        return line;
    }
}
