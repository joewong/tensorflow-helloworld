package dev.joewong.tensorflow.helloworld;

import org.tensorflow.*;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/***
 * This is a rewrite of the neural network mentioned here:
 * http://www.cs.bham.ac.uk/~jxb/INC/nn.html
 */

public class HelloTensorFlow {

    private static final int EPOCHS = 6000;
    private static final float ETA_VALUE = 0.5f;
    private static final float ALPHA_VALUE = 0.9f;
    private static final int INPUT_NEURONS = 2;
    private static final int HIDDEN_NEURONS = 8;
    private static final int OUTPUT_NEURONS = 1;
    private static final Shape INPUT_SHAPE = Shape.make(INPUT_NEURONS, 1);
    private static final Shape HIDDEN_SHAPE = Shape.make(HIDDEN_NEURONS, 1);

    private Graph graph;
    private Session session;
    private Output inputOutput;
    private Output expectedOutput;
    private Output inputHiddenWeightOutput;
    private Output inputHiddenWeightBiasOutput;
    private Output hiddenOutputWeightOutput;
    private Output hiddenOutputWeightBiasOutput;
    private Output deltaInputHiddenWeightBiasOutput;
    private Output deltaInputHiddenWeightOutput;
    private Output deltaHiddenOutputWeightBiasOutput;
    private Output deltaHiddenOutputWeightOutput;

    private float[] predictionValues = NNUtil.createFloatArray(OUTPUT_NEURONS);
    private float[][] inputHiddenWeightValues = NNUtil.create2DFloatArray(HIDDEN_NEURONS, INPUT_NEURONS);
    private float[] inputHiddenWeightBiasValues = NNUtil.createFloatArray(HIDDEN_NEURONS);
    private float[][] hiddenOutputWeightValues = NNUtil.create2DFloatArray(OUTPUT_NEURONS, HIDDEN_NEURONS);
    private float[] hiddenOutputWeightBiasValues = NNUtil.createFloatArray(OUTPUT_NEURONS, 0.0f);
    private float[][] deltaInputHiddenWeightValues = NNUtil.create2DFloatArray(HIDDEN_NEURONS, INPUT_NEURONS, 0.0f);
    private float[] deltaInputHiddenWeightBiasValues = NNUtil.createFloatArray(HIDDEN_NEURONS, 0.0f);
    private float[] deltaHiddenOutputWeightBiasValues = NNUtil.createFloatArray(OUTPUT_NEURONS);
    private float[][] deltaHiddenOutputWeightValues = NNUtil.create2DFloatArray(OUTPUT_NEURONS, HIDDEN_NEURONS, 0.0f);

    private DisplayFloatValues displayFloatValues = new DisplayFloatValues();
    
    public HelloTensorFlow() {
        initGraph();
        addGraphVariables();
        addGraphOperations();
    }

    public void initGraph() {
        graph = new Graph();
        session = new Session(graph);
    }

    public void addGraphVariables() {
        inputOutput = Ops.placeholder(graph, "input_values", DataType.FLOAT);
        expectedOutput = Ops.placeholder(graph, "expected_output", DataType.FLOAT);
        inputHiddenWeightBiasOutput = Ops.variable(graph, "input_hidden_bias", DataType.FLOAT, Shape.make(1));
        deltaInputHiddenWeightBiasOutput = Ops.variable(graph, "delta_input_hidden_bias", DataType.FLOAT, Shape.make(1));
        hiddenOutputWeightBiasOutput = Ops.variable(graph, "hidden_output_weight_bias", DataType.FLOAT, Shape.make(1));
        deltaHiddenOutputWeightBiasOutput = Ops.variable(graph, "delta_hidden_output_weight_bias", DataType.FLOAT, Shape.make(1));
        deltaInputHiddenWeightOutput = Ops.variable(graph, "delta_input_hidden_weight", DataType.FLOAT, INPUT_SHAPE);
        deltaHiddenOutputWeightOutput = Ops.variable(graph, "delta_hidden_output_weight", DataType.FLOAT, HIDDEN_SHAPE);
        inputHiddenWeightOutput = Ops.variable(graph, "input_synapse", DataType.FLOAT, INPUT_SHAPE);
        hiddenOutputWeightOutput = Ops.variable(graph, "output_synapse", DataType.FLOAT, HIDDEN_SHAPE);
    }

    public void addGraphOperations() {
        Output REDUCE_DIMENSION = Ops.constant(graph, "reduce_dimension", Tensor.create(new int[]{1}));
        Output MINUS_ONE = Ops.constant(graph, "minus_one", Tensor.create(-1.0F));
        Output ONE_FLOAT = Ops.constant(graph, "one", Tensor.create(1.0F));
        Output ONE_INT = Ops.constant(graph, "one_int", Tensor.create(1));
        Output ALPHA = Ops.constant(graph, "alpha", Tensor.create(ALPHA_VALUE));
        Output ETA = Ops.constant(graph, "eta", Tensor.create(ETA_VALUE));
        
        Output mulInputByInputSynapseCalc = Ops.multiply(graph, "mul_input_by_input_hidden", inputOutput, inputHiddenWeightOutput);

        Output sumHiddenACalc = Ops.sum(graph, "sum_input", DataType.FLOAT, mulInputByInputSynapseCalc, REDUCE_DIMENSION);

        Output sumHiddenUpdateCalc = Ops.add(graph, "add_input_hidden_bias", sumHiddenACalc, inputHiddenWeightBiasOutput);
    
        Output mulMinusOneInputCalc = Ops.multiply(graph, "mul_minus_one", sumHiddenUpdateCalc, MINUS_ONE);

        Output expMinusSumHiddenCalc = Ops.exp(graph, "exp_input", mulMinusOneInputCalc);

        Output onePlusExpMinusSumHiddenCalc = Ops.add(graph, "add_one_input", expMinusSumHiddenCalc, ONE_FLOAT);

        Output hiddenUpdateCalc = Ops.divide(graph, "hidden_update", ONE_FLOAT, onePlusExpMinusSumHiddenCalc);

        Ops.expandDims(graph, "increase_hidden_output_bias_dim", hiddenOutputWeightBiasOutput, ONE_INT);

        Output mulHiddenByHiddenOutputWeightCalc = Ops.multiply(graph, "mul_hidden_by_hidden_output_weight", hiddenUpdateCalc, hiddenOutputWeightOutput);

        Output sumOutputCalc = Ops.sum(graph, "sum_output", DataType.FLOAT, mulHiddenByHiddenOutputWeightCalc, REDUCE_DIMENSION);

        Output sumOutputUpdateCalc = Ops.add(graph, "add_hidden_output_bias", sumOutputCalc, hiddenOutputWeightBiasOutput);

        Output mulHiddenMinusOneInputCalc = Ops.multiply(graph, "mul_hidden_minus_one", sumOutputUpdateCalc, MINUS_ONE);

        Output expHiddenCalc = Ops.exp(graph, "exp_hidden", mulHiddenMinusOneInputCalc);
        
        Output addOneHiddenCalc = Ops.add(graph, "add_one_hidden", expHiddenCalc, ONE_FLOAT);

        Output predictionUpdateCalc = Ops.divide(graph, "prediction", ONE_FLOAT, addOneHiddenCalc);

        Output subOneFromPredictionCalc = Ops.subtract(graph, "sub_one_from_output", ONE_FLOAT, predictionUpdateCalc);

        Output subInputFromPredictionCalc = Ops.subtract(graph, "sub_target_from_output", expectedOutput, predictionUpdateCalc);

        Output mulInputMinusPredictionByPredictionCalc = Ops.multiply(graph, "mul_target_minus_output_by_output", subInputFromPredictionCalc, predictionUpdateCalc);

        Output deltaOutputUpdateCalc = Ops.multiply(graph, "delta_output_update", mulInputMinusPredictionByPredictionCalc, subOneFromPredictionCalc);

        Output bpeMulWeightHiddenOutputByDeltaOutputCalc = Ops.multiply(graph, "mul_weight_hidden_output_by_delta_output", hiddenOutputWeightOutput, deltaOutputUpdateCalc);

        Output sumDeltaOutputWeightUpdateCalc = Ops.sum(graph, "sum_multiply", DataType.FLOAT, bpeMulWeightHiddenOutputByDeltaOutputCalc, REDUCE_DIMENSION);

        Output oneMinusHiddenCalc = Ops.subtract(graph, "sub_one_from_hidden", ONE_FLOAT, hiddenUpdateCalc);

        Output mulHiddenBySumOneMinusHiddenCalc = Ops.multiply(graph, "mul_hidden_by_sum_multiply", oneMinusHiddenCalc, hiddenUpdateCalc);

        Output deltaHiddenUpdateCalc = Ops.multiply(graph, "delta_hidden_update", sumDeltaOutputWeightUpdateCalc, mulHiddenBySumOneMinusHiddenCalc);

        Output mulAlphaByDeltaWeightInputHidden_forBias_Calc = Ops.multiply(graph, "mul_alpha_by_delta_weight_input_hidden", ALPHA, deltaInputHiddenWeightBiasOutput);

        Output mulEtaByDeltaHidden_forBias_Calc = Ops.multiply(graph, "mul_eta_by_delta_hidden", ETA, deltaHiddenUpdateCalc);

        Output deltaInputHiddenWeightBiasUpdateCalc = Ops.add(graph, "delta_input_hidden_weight_bias_update", mulAlphaByDeltaWeightInputHidden_forBias_Calc, mulEtaByDeltaHidden_forBias_Calc);

        Ops.add(graph, "input_hidden_weight_bias_update", inputHiddenWeightBiasOutput, deltaInputHiddenWeightBiasUpdateCalc);

        Output mulAlphaByDeltaWeightInputHiddenCalc = Ops.multiply(graph, "mul_alpha_by_delta_weight_input_hidden_result", ALPHA, deltaInputHiddenWeightOutput);

        Output mulEtaByInputCalc = Ops.multiply(graph, "mul_eta_by_input", ETA, inputOutput);

        Output increaseDeltaHiddenDimensionBy1Calc = Ops.expandDims(graph, "increase_delta_hidden_dim", deltaHiddenUpdateCalc, ONE_INT);
        
        Output mulEtaByInputByDeltaHiddenCalc = Ops.multiply(graph, "mul_eta_by_input_by_delta_hidden", mulEtaByInputCalc, increaseDeltaHiddenDimensionBy1Calc);

        Output deltaInputHiddenWeightUpdateCalc = Ops.add(graph, "delta_weight_input_hidden_update", mulAlphaByDeltaWeightInputHiddenCalc, mulEtaByInputByDeltaHiddenCalc);
    
        Ops.add(graph, "input_hidden_weight_update",inputHiddenWeightOutput, deltaInputHiddenWeightUpdateCalc);

        Output mulAlphaAndDeltaHiddenOutputBias_forDeltaHiddenOutputWeightBias_Calc = Ops.multiply(graph, "mul_alpha_and_delta_hidden_output_bias", ETA, deltaOutputUpdateCalc);

        Output mulEtaByDeltaOutput_forDeltaHiddenOutputWeightBias_Calc = Ops.multiply(graph, "mul_eta_by_delta_output", ALPHA, deltaHiddenOutputWeightBiasOutput);

        Output deltaHiddenOutputWeightBiasUpdateCalc = Ops.add(graph, "delta_hidden_output_weight_bias_update", mulAlphaAndDeltaHiddenOutputBias_forDeltaHiddenOutputWeightBias_Calc, mulEtaByDeltaOutput_forDeltaHiddenOutputWeightBias_Calc);

        Ops.add(graph, "hidden_output_weight_bias_update", hiddenOutputWeightBiasOutput, deltaHiddenOutputWeightBiasUpdateCalc);

        Output mulAlphaByDeltaWeightHiddenOutputCalc = Ops.multiply(graph, "mul_alpha_by_delta_weight_hidden_output", ALPHA, deltaHiddenOutputWeightOutput);
        
        Output mulEtaByHiddenCalc = Ops.multiply(graph, "mul_eta_by_hidden", ETA, hiddenUpdateCalc);
                
        Output mulEtaByHiddenByDeltaOutputCalc = Ops.multiply(graph, "mul_eta_by_hidden_by_delta_output", mulEtaByHiddenCalc, deltaOutputUpdateCalc);

        Output deltaHiddenOutputWeightUpdateCalc = Ops.add(graph, "delta_hidden_output_weight_update", mulEtaByHiddenByDeltaOutputCalc, mulAlphaByDeltaWeightHiddenOutputCalc);

        Ops.add(graph, "weight_hidden_output_update", hiddenOutputWeightOutput, deltaHiddenOutputWeightUpdateCalc);
    }

    public void run() {
        Float[][] inputs = XorDataSet.getDataSet();
        Float[] input;
        Float expected;

        for (int i = 0; i<EPOCHS; i++) {
            Collections.shuffle(Arrays.asList(inputs));
            for (int j=0; j<inputs.length; j++) {
                input = inputs[j];
                float[] row = {input[0], input[1]};
                expected = input[2];

                System.out.println("================");
                System.out.println("EPOCH: " + i);
                System.out.println("Data set item: " + j);
                System.out.println("================");

                System.out.println("\nInput:");
                System.out.println(displayFloatValues.displayString(row));

                System.out.println("\nExpected:");
                System.out.println(displayFloatValues.displayString(expected));

                displayGraphRun(runGraph(row, expected));
            }
        }
    }

    private List<Tensor<?>> runGraph(float[] trainRowValues,float expectedValues) {
        return session.runner()
        .feed(inputOutput, Tensor.create(trainRowValues))
        .feed(expectedOutput, Tensor.create(expectedValues))
        .feed(inputHiddenWeightOutput, Tensor.create(inputHiddenWeightValues))
        .feed(inputHiddenWeightBiasOutput, Tensor.create(inputHiddenWeightBiasValues))
        .feed(hiddenOutputWeightOutput, Tensor.create(hiddenOutputWeightValues))
        .feed(hiddenOutputWeightBiasOutput, Tensor.create(hiddenOutputWeightBiasValues))
        .feed(deltaInputHiddenWeightBiasOutput, Tensor.create(deltaInputHiddenWeightBiasValues))
        .feed(deltaInputHiddenWeightOutput, Tensor.create(deltaInputHiddenWeightValues))
        .feed(deltaHiddenOutputWeightOutput, Tensor.create(deltaHiddenOutputWeightValues))
        .feed(deltaHiddenOutputWeightBiasOutput, Tensor.create(deltaHiddenOutputWeightBiasValues))
        .fetch("prediction")
        .fetch("delta_input_hidden_weight_bias_update")
        .fetch("input_hidden_weight_bias_update")
        .fetch("delta_weight_input_hidden_update")
        .fetch("input_hidden_weight_update")
        .fetch("delta_hidden_output_weight_bias_update")
        .fetch("hidden_output_weight_bias_update")
        .fetch("delta_hidden_output_weight_update")
        .fetch("weight_hidden_output_update")
        .run();
    }

    private void displayGraphRun(List<Tensor<?>> t) {
        t.get(0).copyTo(predictionValues);
        t.get(1).copyTo(deltaInputHiddenWeightBiasValues);     
        t.get(2).copyTo(inputHiddenWeightBiasValues);          
        t.get(3).copyTo(deltaInputHiddenWeightValues);         
        t.get(4).copyTo(inputHiddenWeightValues);              
        t.get(5).copyTo(deltaHiddenOutputWeightBiasValues);    
        t.get(6).copyTo(hiddenOutputWeightBiasValues);         
        t.get(7).copyTo(deltaHiddenOutputWeightValues);        
        t.get(8).copyTo(hiddenOutputWeightValues);
        System.out.println("Prediction: ");
        System.out.println(displayFloatValues.displayString(predictionValues) + "\n");
    }
}