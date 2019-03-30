package dev.joewong.tensorflow.helloworld;

import org.tensorflow.*;

import java.util.Random;

public class Ops {

    public static Output setInitialFloatValues(Graph graph, String name, float[][] values, Output assignee) {
        return assign(graph, name + "_initial_values",assignee, getFloatAssignValues(graph, name + "_values", values));
    }

    public static float getRand() {
        Random r = new Random();
        return r.nextFloat();
    }

    public static Output getFloatAssignValues(Graph graph, String name, float[][] values) {
        return Ops.constant(graph, name,Tensor.create(values));
    }

    public static Output constant(Graph graph, String name, Tensor t) {
        return graph.opBuilder("Const", name)
                .setAttr("dtype", t.dataType())
                .setAttr("value", t)
                .build().output(0);
    }

    public static Output sigmoid(Graph graph, String name, Output input) {
        return graph.opBuilder("Sigmoid", name)
                .addInput(input)
                .build()
                .output(0);
    }

    public static Output exp(Graph graph, String name, Output input) {
        return graph.opBuilder("Exp", name)
                .addInput(input)
                .build()
                .output(0);
    }

    public static Output placeholder(Graph graph, String name, DataType dataType) {
        return graph.opBuilder("Placeholder", name)
                .setAttr("dtype", dataType)
                .build().output(0);
    }

    public static Output variable(Graph graph, String name, DataType dataType, Shape shape) {
        return graph.opBuilder("Variable", name)
                .setAttr("dtype", dataType)
                .setAttr("shape", shape)
                .build().output(0);
    }

    public static Output sum(Graph graph, String name, DataType dataType, Output operand1, Output operand2) {
        return graph.opBuilder("Sum", name)
                .setAttr("T", dataType)
                .setAttr("Tidx", DataType.INT32)
                .addInput(operand1)
                .addInput(operand2)
                .build().output(0);
    }

    public static Output matMul(Graph graph, String name, Output operand1, Output operand2, Boolean transposeA, Boolean transposeB) {
        return graph.opBuilder("MatMul", name)
                .addInput(operand1)
                .addInput(operand2)
                .setAttr("transpose_a",transposeA)
                .setAttr("transpose_b",transposeB)
                .build().output(0);
    }

    public static Output batchMatMul(Graph graph, String name, Output operand1, Output operand2) {
        return graph.opBuilder("BatchMatMul", name)
                .addInput(operand1)
                .addInput(operand2)
                .build().output(0);
    }

    public static Output twoOperandOp(Graph graph, String operation, String name, Output operand1, Output operand2) {
        return graph.opBuilder(operation, name)
                .addInput(operand1)
                .addInput(operand2)
                .build()
                .output(0);
    }

    public static Output assign(Graph graph, String name, Output operand1, Output operand2) {
        return twoOperandOp(graph,"Assign", name, operand1, operand2);
    }

    public static Output assignVar(Graph graph, String name, Output operand1, Output operand2) {
        return twoOperandOp(graph,"AssignVariableOp", name, operand1, operand2);
    }

    public static Output sigmoidGrad(Graph graph, String name, Output operand1, Output operand2) {
        return twoOperandOp(graph,"SigmoidGrad", name, operand1, operand2);
    }

    public static Output subtract(Graph graph, String name, Output operand1, Output operand2) {
        return twoOperandOp(graph,"Sub", name, operand1, operand2);
    }

    public static Output transpose(Graph graph, String name, Output operand1, Output operand2) {
        return twoOperandOp(graph,"Transpose", name, operand1, operand2);
    }

    public static Output reshape(Graph graph, String name, Output operand1, Output operand2) {
        return twoOperandOp(graph,"Reshape", name, operand1, operand2);
    }

    public static Output add(Graph graph, String name, Output operand1, Output operand2) {
        return twoOperandOp(graph,"Add", name, operand1, operand2);
    }

    public static Output assignAdd(Graph graph, String name, Output operand1, Output operand2) {
        return twoOperandOp(graph,"AssignAdd", name, operand1, operand2);
    }

    public static Output multiply(Graph graph, String name, Output operand1, Output operand2) {
        return twoOperandOp(graph,"Mul", name, operand1, operand2);
    }

    public static Output divide(Graph graph, String name, Output operand1, Output operand2) {
        return twoOperandOp(graph,"Div", name, operand1, operand2);
    }

    public static Output expandDims(Graph graph, String name, Output operand1, Output operand2) {
        return twoOperandOp(graph,"ExpandDims", name, operand1, operand2);
    }
}
