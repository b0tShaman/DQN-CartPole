package src.DQN;

import src.Common.InputNeuron;
import src.Common.Neuron;
import src.Common.NeuronBase;

import javax.swing.*;
import java.io.*;
import java.util.*;

import static src.Common.Neuron.BATCH;

public class DQN {
    static boolean train = false;
    static double Length_Of_Stick = 0.326;// length of the stick on cart pole
    static double Mass_Of_Cart = 0.711; // mass of the cart
    static double Mass_Of_Stick = 0.209; // mass of the stick
    static double g = 9.8; // acceleration due to gravity
    static double initial_angle_of_stick = 7 * (Math.PI/180); // 0 - upright(balanced), 30 = 30 degrees to right, 90 = horizontal
    static int fps = 30; // frames per second for cart pole animation
    public static String Filepath = "./src/DQN/"; // path to store weights of Neural Networks

    static int M = 200;
    int maxEpisodeRewardRequired = 0;
    static int EPISODES = 30;
    static int EPOCH = 1;
    double discount = 0.99;
    int actionSpace = 2;

    double [][]Output_Batch ;
    double error;

    List<NeuronBase> q_InputLayer = new ArrayList<>();
    List<NeuronBase> q_L1_Layer = new ArrayList<>();
    List<NeuronBase> q_Output_Layer = new ArrayList<>();

    List<NeuronBase> target_InputLayer = new ArrayList<>();
    List<NeuronBase> target_L1_Layer = new ArrayList<>();
    List<NeuronBase> target_Output_Layer = new ArrayList<>();

    public static void main(String[] args) throws Exception{
        if (args.length > 0) {
            // Convert the argument to boolean
            boolean flag = Boolean.parseBoolean(args[0]);
            if (flag) {
                train = true;
            }
        }
        DQN dqn = new DQN();
        dqn.DeepQNetwork();
    }

    public void DeepQNetwork() throws IOException, InterruptedException {
        List<Double> avgRewardList = new ArrayList<>();
        long seed = -98;
        // Initialize Q network
        for (int j1 = 0; j1 < 4; j1++) {
            q_L1_Layer.add(new Neuron()
                    .setLearning_Rate(Math.pow(10,-1))
                    .setSeed(seed)
                    .setActivation(Neuron.Activation_Function.LINEAR)
                    .setWeightsFileName("Q_Network")
                    .setWeightsFileIndex(j1)
                    .setWeightInitializationAlgorithm(Neuron.Weight_Initialization.constant)
                    .setOptimizerAlgorithm(Neuron.Optimization_Algorithm.Gradient_Descent)
                    .build());
        }

        for (int j1 = 0; j1 < actionSpace; j1++) {
            q_Output_Layer.add(new Neuron()
                    .setLearning_Rate(Math.pow(10,-1))
                    .setSeed(seed)
                    .setActivation(Neuron.Activation_Function.LINEAR)
                    .setWeightsFileName("Q_Network")
                    .setWeightsFileIndex(j1)
                    .setWeightInitializationAlgorithm(Neuron.Weight_Initialization.constant)
                    .setOptimizerAlgorithm(Neuron.Optimization_Algorithm.Gradient_Descent)
                    .build());
        }

        // Initialize target network
        for (int j1 = 0; j1 < 4; j1++) {
            target_L1_Layer.add(new Neuron()
                    .setLearning_Rate(Math.pow(10,-1))
                    .setSeed(seed)
                    .setActivation(Neuron.Activation_Function.LINEAR)
                    .setWeightsFileName("Target_Network")
                    .setWeightsFileIndex(j1)
                    .setWeightInitializationAlgorithm(Neuron.Weight_Initialization.constant)
                    .setOptimizerAlgorithm(Neuron.Optimization_Algorithm.Gradient_Descent)
                    .build());
        }

        for (int j1 = 0; j1 < actionSpace; j1++) {
            target_Output_Layer.add(new Neuron()
                    .setLearning_Rate(Math.pow(10,-1))
                    .setSeed(seed)
                    .setActivation(Neuron.Activation_Function.LINEAR)
                    .setWeightsFileName("Target_Network")
                    .setWeightsFileIndex(j1)
                    .setWeightInitializationAlgorithm(Neuron.Weight_Initialization.constant)
                    .setOptimizerAlgorithm(Neuron.Optimization_Algorithm.Gradient_Descent)
                    .build());
        }

        for(int v=0 ; v < M && train; v++) {
            File trainingDataFile = new File(Filepath + "trainingData.csv");
            if (trainingDataFile.exists()) {
                System.out.println("trainingDataFile.delete() - " + v + " " + trainingDataFile.delete());
            }
            double episodeReward = 0;
            // createTrainingData
            Environment_CartPole environment = new Environment_CartPole(0.17);
            Environment_CartPole.State curr_state = environment.getState();
            Environment_CartPole.State new_state;

            StateSpaceQuantization stateSpaceQuantization;
            boolean verify = true;

            for (double i = 0; i < EPISODES; i = i + 0.02) {
                stateSpaceQuantization = StateSpaceQuantization.getBox1(curr_state);
                List<Double> curr_state_NN_Input = currStateToInput(curr_state);
                feedForward_Q_Network(curr_state_NN_Input, 0);

                double[] q_Outputs = new double[actionSpace];
                for (int j1 = 0; j1 < actionSpace; j1++) {
                    q_Outputs[j1] = q_Output_Layer.get(j1).getOutput();
                }
                double Q_at = sampleAction(q_Outputs);

                new_state = environment.getNewStateAndReward(Q_at);
                stateSpaceQuantization = StateSpaceQuantization.getBox1(new_state);
                verify = StateSpaceQuantization.verify(stateSpaceQuantization);

                double reward = new_state.reward;
                episodeReward = episodeReward + reward;
                List<Double> new_state_NN_Input = currStateToInput(new_state);

                StoreTrainingData(curr_state_NN_Input, Q_at, reward, new_state_NN_Input);

                curr_state = new_state;
                clearCache();
                if (!verify) {
                    System.out.println("EPISODE END !!!!!, time = " + (new_state.time) + ", x = " + (new_state.x) + ", theta_degree = " + (new_state.theta_degree));
                    break;
                }
            }

            //System.out.println("episodeReward " + episodeReward);
            avgRewardList.add(episodeReward);
            List<List<String>> trainingSet = new ArrayList<>();
            try (BufferedReader br = new BufferedReader(new FileReader(Filepath + "trainingData.csv"))) {
                String line;
                while ((line = br.readLine()) != null) {
                    String[] values = line.split(",");
                    trainingSet.add(Arrays.asList(values));
                }
            }

            BATCH = trainingSet.size();
            //BATCH = trainingSet.size()/1000;
            reInitBatchMemory();
            Output_Batch = new double[BATCH][];

            // Training Q Network with the help of Target Network
            int epoch = 0;
            while (train && (epoch++ < EPOCH)) {
                Collections.shuffle(trainingSet);
                for (int index = 0; index < trainingSet.size(); index++) {
                    try {
                        List<String> data = trainingSet.get(index);

                        int batchIndex = index % BATCH;

                        feedForward_Q_Network(stringToList(data.get(0)), batchIndex);
                        feedForward_Target_Network(stringToList(data.get(3)), batchIndex);

                        double[] q_Outputs = new double[actionSpace];
                        for (int j1 = 0; j1 < actionSpace; j1++) {
                            q_Outputs[j1] = q_Output_Layer.get(j1).getOutput();
                        }

                        double[] target_Outputs = new double[actionSpace];
                        for (int j1 = 0; j1 < actionSpace; j1++) {
                            target_Outputs[j1] = target_Output_Layer.get(j1).getOutput();
                        }

                        double Q_at = StateSpaceQuantization.max(q_Outputs);
                        double Q_atNext = StateSpaceQuantization.max(target_Outputs);
                        int j = StateSpaceQuantization.argMax(q_Outputs);

                        double reward = Double.parseDouble(data.get(2));

                        if (reward == -1)
                            error = reward - Q_at;
                        else
                            error = ((reward) + discount * Q_atNext) - Q_at;

                        Output_Batch[batchIndex] = formHotEncodedVector(error, j);

                        clearCache();

                        if ((index + 1) % BATCH == 0) {
                            for (int batchIndex1 = 0; batchIndex1 < Output_Batch.length; batchIndex1++) {
                                feedback_Q_Network(batchIndex1, Output_Batch[batchIndex1]);
                            }
                        }

                    } catch (Exception e) {
                        System.out.println("Exception:" + e);
                    }
                }
                copyWeights();
            }
            double avgEpisodeReward = 0;
            avgEpisodeReward = (findMean(avgRewardList, 4));
            if( v > 4) {
                System.out.println("avgEpisodeReward == " + avgEpisodeReward);
                if (avgEpisodeReward >= maxEpisodeRewardRequired) break;
            }
        }
        // Store Neural Network weights
        if( train) {
            File statsFile = new File(Filepath + "Q_Network.csv");
            if (statsFile.exists()) {
                System.out.println("statsFile.delete() - " + statsFile.delete());
            }
            File statsFile2 = new File(Filepath + "Target_Network.csv");
            if (statsFile2.exists()) {
                System.out.println("statsFile2.delete() - " + statsFile2.delete());
            }
            Store_ANN_Weights();
        }

        // Testing
        {
            Environment_CartPole environment = new Environment_CartPole(initial_angle_of_stick);
            Environment_CartPole.State curr_state = environment.getState();
            Environment_CartPole.State new_state ;

            JFrame f = new JFrame("CartPole");
            CartPole p = new CartPole(Length_Of_Stick);
            p.setState(curr_state);
            f.add(p);
            f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            f.pack();
            f.setVisible(true);
            new Thread(p).start();

            for (double i = 0; i < 60; i = i + 0.02) {
                System.out.println("state: " +curr_state.toString());

                List<Double> curr_state_NN_Input = currStateToInput(curr_state);
                feedForward_Q_Network(curr_state_NN_Input,0);

                double[] q_Outputs = new double[actionSpace];
                for (int j1 = 0; j1 < actionSpace; j1++) {
                    q_Outputs[j1] = q_Output_Layer.get(j1).getOutput();
                }

                double Q_at = sampleAction(q_Outputs);
                System.out.println("action: " +Q_at);
                new_state = environment.getNewStateAndReward(Q_at);

                curr_state = new_state;
                p.setState(curr_state);

                clearCache();
                Thread.sleep(1000/fps);
            }
        }
        Thread.sleep(2000);
    }

    public void feedForward_Q_Network(List<Double> input, int batchIndex) throws IOException {
        for (Double d : input) {
            q_InputLayer.add(new InputNeuron(d));
        }

        if (!q_L1_Layer.isEmpty()){
            for (NeuronBase neuron : q_L1_Layer) {
                neuron.feedForward(q_InputLayer,batchIndex );
            }
        }

        if (!q_Output_Layer.isEmpty()) {
            for (NeuronBase neuron : q_Output_Layer) {
                neuron.feedForward(q_L1_Layer,batchIndex );
            }
        }
    }

    public void feedForward_Target_Network(List<Double> input, int batchIndex) throws IOException {
        for (Double integer : input) {
            target_InputLayer.add(new InputNeuron(integer));
        }

        if (!target_L1_Layer.isEmpty()){
            for (NeuronBase neuron : target_L1_Layer) {
                neuron.feedForward(target_InputLayer,batchIndex );
            }
        }

        if (!target_Output_Layer.isEmpty()) {
            for (NeuronBase neuron : target_Output_Layer) {
                neuron.feedForward(target_L1_Layer,batchIndex );
            }
        }
    }

    public void feedback_Q_Network(int batchIndex, double[] error){
        int q_action = getQ_Action(error);

        q_Output_Layer.get(q_action).setError_Next(error[q_action]);

        if (!q_Output_Layer.isEmpty()) {
            for (NeuronBase neuronBase : q_Output_Layer) {
                neuronBase.feedBack(batchIndex, 0);
            }
        }

        if (!q_L1_Layer.isEmpty()) {
            for (NeuronBase neuronBase : q_L1_Layer) {
                neuronBase.feedBack(batchIndex, 0);
            }
        }
    }

    public void clearCache(){
        q_InputLayer.clear();
        target_InputLayer.clear();
    }

    public List<Double> currStateToInput(Environment_CartPole.State state){
        List<Double> input = new ArrayList<>();
        input.add(state.theta);
        input.add(state.theta_dot);
        input.add(state.x);
        input.add(state.x_dot);
        return input;
    }

    public double sampleAction(double[] q_Outputs){
        return -10 + 20*StateSpaceQuantization.argMax(q_Outputs);
    }

    public static int getQ_Action(double[] Q) {
        for (int t = 0; t < Q.length; t++) {
            if (Q[t] != 0) {
                return t;
            }
        }
        return Q.length -1;
    }

    public void reInitBatchMemory(){
        if (!q_L1_Layer.isEmpty()){
            for (NeuronBase neuron : q_L1_Layer) {
                neuron.reInitializeBatchMemory();
            }
        }

        if (!q_Output_Layer.isEmpty()) {
            for (NeuronBase neuron : q_Output_Layer) {
                neuron.reInitializeBatchMemory();
            }
        }

        if (!target_L1_Layer.isEmpty()){
            for (NeuronBase neuron : target_L1_Layer) {
                neuron.reInitializeBatchMemory();
            }
        }

        if (!target_Output_Layer.isEmpty()) {
            for (NeuronBase neuron : target_Output_Layer) {
                neuron.reInitializeBatchMemory();
            }
        }
    }

    public List<Double> stringToList(String str){
        List<Double> integers = new ArrayList<>();
        String[] strings = str.split(";");
        for (int i=0; i< strings.length;i++){
            integers.add(Double.parseDouble(strings[i]));
        }
        return integers;
    }

    public double[] formHotEncodedVector(double yi, double j){
        double[] Y_Encoded = new double[actionSpace];
        for (int i=0; i< Y_Encoded.length; i++){
            if( i == j){
                Y_Encoded[i] = yi;
            }
            else Y_Encoded[i] = 0;
        }
        return Y_Encoded;
    }

    public void copyWeights(){
        for (int i=0 ; i< q_L1_Layer.size(); i ++){
            target_L1_Layer.get(i).setWeights(q_L1_Layer.get(i).getWeights());
        }

        for (int i=0 ; i< q_Output_Layer.size(); i ++){
            target_Output_Layer.get(i).setWeights(q_Output_Layer.get(i).getWeights());
        }
    }

    public void Store_ANN_Weights(){
        for (NeuronBase neuronL4 : q_L1_Layer) {
            neuronL4.Memorize();
        }

        for (NeuronBase neuronL4 : target_L1_Layer) {
            neuronL4.Memorize();
        }

        for (NeuronBase neuronL4 : q_Output_Layer) {
            neuronL4.Memorize();
        }

        for (NeuronBase neuronL4 : target_Output_Layer) {
            neuronL4.Memorize();
        }
    }

    public void StoreTrainingData(List<Double> s_t, double at, double reward, List<Double> s_tNext){
        try {
            File statsFile = new File(Filepath + "trainingData.csv");
            if (statsFile.exists()) {
            } else {
                FileWriter out = new FileWriter(statsFile);
                out.flush();
                out.close();
            }

            if (statsFile.exists()) {
                FileWriter buf = new FileWriter(Filepath + "trainingData.csv", true);
                for (Double integer : s_t) {
                    buf.append(String.valueOf(integer));
                    buf.append(";");
                }
                buf.append(",");
                buf.append(String.valueOf(at));
                buf.append(",");
                buf.append(String.valueOf(reward));
                buf.append(",");
                for (Double integer : s_tNext) {
                    buf.append(String.valueOf(integer));
                    buf.append(";");
                }
                buf.append("\n");
                buf.flush();
                buf.close();
            } else {
                System.out.println("StoreTrainingData FAIL 3 NO FILE");
            }
        }
        catch (Exception ex){
            System.out.println("StoreTrainingData FAIL 4"  + ex.getMessage());
        }
    }

    public double findMean (List<Double> X, int n) {
        double sum =0;
        double mean =0;
        // calculate mean and std of all columns
        for (int i = X.size()-1, k = 0; k < n && i>-1; i--, k++) {
            sum = (sum + X.get(i));
        }

        mean = sum/n;
        return mean;
    }
}
