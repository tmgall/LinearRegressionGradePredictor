// imports
import java.io.FileNotFoundException;
import java.io.File;
import java.util.Random;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Predictor  {

    public static final int MIN_SCORE = 0;
    public static final int MAX_SCORE = 20;

    public static void main(String[] args) throws FileNotFoundException {
        // import data, make it into an array, then randomize the order
        File f = new File("student-mat.csv");
        double[][] data = shuffleData(parseData(f));

        // useful values
        int setSize = data.length;
        int trainingSize = (int) (Math.round(setSize * 0.7));
        int features = data[0].length;

        // partition data into sets 
        double[][] trainingSet = getInputsAndBias(data, 0, trainingSize);
        double[] trainingLabels = getOutputs(data, 0, trainingSize);
        double[][] testSet = getInputsAndBias(data, trainingSize, setSize);
        double[] testLabels = getOutputs(data, trainingSize, setSize);

        // train data to find optimal theta
        double[][] theta = gradientDescent(trainingSet, trainingLabels, new double[features][1], 0.005, 5000, false);
        
        // display results of training
        double cost = cost(theta, testSet, testLabels);
        testTheta(testSet, testLabels, theta);
        System.out.println(cost);
    }

    // method to display testing results

    public static void testTheta(double[][] testSet, double[] testLabels, double[][] theta) {
        int numberOfTests = 10;
        for (int i = 0; i < numberOfTests; i++) {
            double prediction = round(predict(theta, testSet[i]), 1);
            System.out.println("Guess: " + prediction + " Actual: " + testLabels[i]);
        }
    }

    // randomize order of data to ensure randomized training and test sets in main

    public static double[][] shuffleData(double[][] data) {
        Random rand = new Random();
        double[][] dataCopy = data;
        int[] indices = new int[data.length];
        for (int i = 0; i < data.length; i++) {
            indices[i] = -1;
        }
        for (int i = 0; i < data.length; i++) {
            boolean used = true;
            while (used) {
                int index = rand.nextInt(395);
                boolean unused = true;
                for (int j = 0; j < data.length; j++) {
                    if (indices[j] == index) {
                        unused = false;
                    }
                }
                if (unused) {
                    indices[i] = index;
                    used = false;
                }
            }
        }
        for (int i = 0; i < data.length; i++) {
            data[i] = dataCopy[indices[i]];
        }
        return data;
    }

    // linear regression methods

    public static double predict(double[][] theta, double[] grades) {
        double prediction = dotVectors(columnToRowVector(theta, 0), grades);
        if (prediction < MIN_SCORE) {
            return MIN_SCORE;
        }
        if (prediction > MAX_SCORE) {
            return MAX_SCORE;
        }
        return prediction;
    }

    public static double[][] gradientDescent(double[][] grades, double[] finalGrades, double[][] theta, double alpha, int iterations, boolean updates) {
        for (int iteration = 0; iteration < iterations; iteration++) {
            double[] gradients = new double[grades[0].length];
            double[][] predictions = dotArrays(grades, theta);
            for (int i = 0; i < grades.length; i++) {
                for (int j = 0; j < gradients.length; j++) {
                    gradients[j] += (predictions[i][0] - finalGrades[i]) * grades[i][j];
                }
            }
            for (int j = 0; j < gradients.length; j++) {
                gradients[j] *= alpha / grades.length;
                theta[j][0] -= gradients[j];
            }
            if (updates) {
                System.out.println(cost(theta, grades, finalGrades));
            }
        }
        return theta;
    }

    public static double cost(double[][] theta, double[][] grades, double[] finalGrades) {
        double[][] predictions = dotArrays(grades, theta);
        double cost = 0;
        for (int i = 0; i < finalGrades.length; i++) {
            double delta = predictions[i][0] - finalGrades[i];
            cost += delta * delta;
        }
        return cost / (2 * finalGrades.length);
    }

    // methods for splitting into data/bias and labels

    public static double[][] getInputsAndBias(double[][] data, int startingIndex, int endingIndex) {
        double[][] inputs = new double[endingIndex - startingIndex][data[0].length];
        for (int i = 0; i < inputs.length; i++) {
            inputs[i][0] = 1;
            for (int j = 1; j < inputs[0].length; j++) {
                inputs[i][j] = data[startingIndex + i][j - 1];
            }
        }
        return inputs;
    }

    public static double[] getOutputs(double[][] data, int startingIndex, int endingIndex) {
        double[] outputs = new double[endingIndex - startingIndex];
        for (int i = 0; i < outputs.length; i++) {
            outputs[i] = data[startingIndex + i][2];
        }
        return outputs;
    }

    // parse data from the attached csv file (does not work on other files)

    public static double[][] parseData(File f) throws FileNotFoundException {
        List<String> data = new ArrayList<>();
        Scanner scan = new Scanner(f);
        while (scan.hasNextLine()) {
            data.add(scan.nextLine());
        }
        scan.close();
        List<List<Double>> scores = new ArrayList<>();
        for (int i = 1; i < data.size(); i++) {
            String[] info = data.get(i).split(";");
            List<Double> newData = new ArrayList<>();
            for (int j = 3; j > 0; j--) {
                if (info[info.length - j].charAt(0) == '\"') {
                    newData.add(Double.valueOf(removeQuotes(info[info.length - j])));
                } else {
                    newData.add(Double.valueOf(info[info.length - j]));
                }
            }
            scores.add(newData);
        }
        return listToArray(scores);
    }

    // linear algebra methods

    public static double[][] dotArrays(double[][] a, double[][] b) {
        double[][] result = new double[a.length][b[0].length];
        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[0].length; j++) {
                result[i][j] = dotVectors(a[i], columnToRowVector(b, j));
            }
        }
        return result;
    }

    public static double dotVectors(double[] a, double[] b) {
        double result = 0;
        for (int i = 0; i < a.length; i++) {
            result += a[i] * b[i];
        }
        return result;
    }

    public static double[] columnToRowVector(double[][] a, int column) {
        double[] result = new double[a.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = a[i][column];
        }
        return result;
    }

    // useful methods

    public static double[][] listToArray(List<List<Double>> data) {
        double[][] arr = new double[data.size()][data.get(0).size()];
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                arr[i][j] = data.get(i).get(j);
            }
        }
        return arr;
    }

    public static void printArray(double[] arr) {
        for (int i = 0; i < arr.length; i++) {
            System.out.print(arr[i] + " ");
        }
        System.out.println();
    }

    public static void printArray(double[][] arr) {
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                System.out.print(arr[i][j] + " ");
            }
            System.out.println();
        }
        System.out.println();
    }

    public static void printList(List<Double> data) {
        for (int i = 0; i < data.size(); i++) {
            System.out.print(data.get(i) + " ");
        }
        System.out.println();
    }

    public static String removeQuotes(String x) {
        return x.substring(1, x.length() - 1);
    }

    public static double round(double number, int precision) {
        int magnitude = (int) Math.pow(10, precision);
        return (double) Math.round(number * magnitude) / magnitude;
    }

    public static double mean(double[] grades) {
        double total = 0;
        for (int i = 0; i < grades.length; i++) {
            total += grades[i];
        }
        return total / grades.length;
    }
}