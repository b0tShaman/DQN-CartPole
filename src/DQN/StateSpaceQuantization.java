package src.DQN;

import java.util.Arrays;

public class StateSpaceQuantization {
    // Q[i][j] = Q[i][j] + learn_rate( Reward + discount * argMax(Q[k]) - Q[i][j]);
    public enum Box {BOX_1, BOX_2, BOX_3}

    public int[] theta_quantized;
    public int[] theta_dot_quantized;
    public int[] x_quantized;
    public int[] x_dot_quantized;

    public static int argMax(double[] Q) {
        int index = Q.length - 1;
        double big = Q[Q.length - 1];
        for (int t = 0; t < Q.length; t++) {
            if (Q[t] > big) {
                big = Q[t];
                index = t;
            }
        }
        return index;
    }

    public static double max(double[] Q) {
        int index = Q.length - 1;
        double big = Q[Q.length - 1];
        for (int t = 0; t < Q.length; t++) {
            if (Q[t] > big) {
                big = Q[t];
                index = t;
            }
        }
        return Q[index];
    }

    public static int[] x_quantization(double x, Box box){
        int[] x_quantized = new int[3];
        if(box.equals(Box.BOX_1)){
            if( (x >= -2.4) && (x < -0.8))
                x_quantized[0] = 1;
            else if ( (x >= -0.8) && (x <= 0.8))
                x_quantized[1] = 1;
            else if ( (x > 0.8) && (x <= 2.4))
                x_quantized[2] = 1;
        }
        if(box.equals(Box.BOX_2)){
            if( (x >= -2.4) && (x < -0.8))
                x_quantized[0] = 1;
            else if ( (x >= -0.8) && (x <= 0.8))
                x_quantized[1] = 1;
            else if ( (x > 0.8) && (x <= 2.4))
                x_quantized[2] = 1;
        }
        if(box.equals(Box.BOX_3)){
            if( (x < -0.5))
                x_quantized[0] = 1;
            else if ( (x >= -0.5) && (x <= 0.5))
                x_quantized[1] = 1;
            else if ( (x > 0.5) )
                x_quantized[2] = 1;
        }
        return x_quantized;
    }

    public static int[] x_dot_quantization(double x, Box box){
        int[] x_quantized = new int[3];
        if(box.equals(Box.BOX_1)){
            if( (x < -0.5))
                x_quantized[0] = 1;
            else if ( (x >= -0.5) && (x <= 0.5))
                x_quantized[1] = 1;
            else if ( (x > 0.5) )
                x_quantized[2] = 1;
        }
        if(box.equals(Box.BOX_2)){
            if( (x < -0.5))
                x_quantized[0] = 1;
            else if ( (x >= -0.5) && (x <= 0.5))
                x_quantized[1] = 1;
            else if ( (x > 0.5) )
                x_quantized[2] = 1;
        }
        if(box.equals(Box.BOX_3)){
            if( (x < -0.5))
                x_quantized[0] = 1;
            else if ( (x >= -0.5) && (x <= 0.5))
                x_quantized[1] = 1;
            else if ( (x > 0.5) )
                x_quantized[2] = 1;
        }
        return x_quantized;
    }

    public static int[] theta_quantization(double x, Box box){
        int[] x_quantized = new int[6];
        if(box.equals(Box.BOX_1)){
            x_quantized = new int[6];
            if( (x >= -12) && (x < -6))
                x_quantized[0] = 1;
            else if ( (x >= -6) && (x < -1))
                x_quantized[1] = 1;
            else if ( (x >= -1) && (x < 0))
                x_quantized[2] = 1;
            else if ( (x >= 0) && (x < 1))
                x_quantized[3] = 1;
            else if ( (x >= 1) && (x < 6))
                x_quantized[4] = 1;
            else if ( (x >= 6) && (x <= 12))
                x_quantized[5] = 1;
        }
        if(box.equals(Box.BOX_2)){
            x_quantized = new int[12];
            if( (x >= -12) && (x <= -6))
                x_quantized[0] = 1;
            else if ( (x > -6) && (x <= -4))
                x_quantized[1] = 1;
            else if ( (x > -4) && (x <= -3))
                x_quantized[2] = 1;
            else if ( (x > -3) && (x <= -2))
                x_quantized[3] = 1;
            else if ( (x > -2) && (x <= -1))
                x_quantized[4] = 1;
            else if ( (x > -1) && (x <= 0))
                x_quantized[5] = 1;
            else if ( (x > 0) && (x <= 1))
                x_quantized[6] = 1;
            else if ( (x > 1) && (x <= 2))
                x_quantized[7] = 1;
            else if ( (x > 2) && (x <= 3))
                x_quantized[8] = 1;
            else if ( (x > 3) && (x <= 4))
                x_quantized[9] = 1;
            else if ( (x > 4) && (x <= 6))
                x_quantized[10] = 1;
            else if ( (x > 6) && (x <= 12))
                x_quantized[11] = 1;
        }
        if(box.equals(Box.BOX_3)){
            x_quantized = new int[6];
            if( (x >= -90) && (x < -60))
                x_quantized[0] = 1;
            else if ( (x >= -60) && (x < -30))
                x_quantized[1] = 1;
            else if ( (x >= -30) && (x < 0))
                x_quantized[2] = 1;
            else if ( (x >= 0) && (x < 30))
                x_quantized[3] = 1;
            else if ( (x >= 30) && (x < 60))
                x_quantized[4] = 1;
            else if ( (x >= 60) && (x <= 90))
                x_quantized[5] = 1;
        }
        return x_quantized;
    }

    public static int[] theta_dot_quantization(double x, Box box){
        int[] x_quantized = new int[3];
        if(box.equals(Box.BOX_1)){
            if( (x < -50))
                x_quantized[0] = 1;
            else if ( (x >= -50) && (x <= 50))
                x_quantized[1] = 1;
            else if ( (x > 50) )
                x_quantized[2] = 1;
        }
        if(box.equals(Box.BOX_2)){
            if( (x < -50))
                x_quantized[0] = 1;
            else if ( (x >= -50) && (x <= 50))
                x_quantized[1] = 1;
            else if ( (x > 50) )
                x_quantized[2] = 1;
        }
        if(box.equals(Box.BOX_3)){
            if( (x < -50))
                x_quantized[0] = 1;
            else if ( (x >= -50) && (x <= 50))
                x_quantized[1] = 1;
            else if ( (x > 50) )
                x_quantized[2] = 1;
        }
        return x_quantized;
    }


    public static StateSpaceQuantization getBox1( Environment_CartPole.State state){
        StateSpaceQuantization stateSpaceQuantization= new StateSpaceQuantization();
        stateSpaceQuantization.theta_quantized = theta_quantization(state.theta_degree,Box.BOX_1);
        stateSpaceQuantization.theta_dot_quantized = theta_dot_quantization(state.theta_dot,Box.BOX_1);
        stateSpaceQuantization.x_quantized = x_quantization(state.x,Box.BOX_1);
        stateSpaceQuantization.x_dot_quantized = x_dot_quantization(state.x_dot,Box.BOX_1);
        return stateSpaceQuantization;
    }

    public static StateSpaceQuantization getBox2( Environment_CartPole.State state){
        StateSpaceQuantization stateSpaceQuantization= new StateSpaceQuantization();
        stateSpaceQuantization.theta_quantized = theta_quantization(state.theta_degree,Box.BOX_2);
        stateSpaceQuantization.theta_dot_quantized = theta_dot_quantization(state.theta_dot,Box.BOX_2);
        stateSpaceQuantization.x_quantized = x_quantization(state.x,Box.BOX_2);
        stateSpaceQuantization.x_dot_quantized = x_dot_quantization(state.x_dot,Box.BOX_2);
        return stateSpaceQuantization;
    }

    public static StateSpaceQuantization getBox3( Environment_CartPole.State state){
        StateSpaceQuantization stateSpaceQuantization= new StateSpaceQuantization();
        stateSpaceQuantization.theta_quantized = theta_quantization(state.theta_degree,Box.BOX_3);
        stateSpaceQuantization.theta_dot_quantized = theta_dot_quantization(state.theta_dot,Box.BOX_3);
        stateSpaceQuantization.x_quantized = x_quantization(state.x,Box.BOX_3);
        stateSpaceQuantization.x_dot_quantized = x_dot_quantization(state.x_dot,Box.BOX_3);
        return stateSpaceQuantization;
    }

    public static boolean verify(StateSpaceQuantization stateSpaceQuantization){
        boolean verify_theta_quantized = false;
        boolean verify_theta_dot_quantized = false;
        boolean verify_x_quantized = false;
        boolean verify_x_dot_quantized = false;
        for (int i=0;i < stateSpaceQuantization.theta_quantized.length;i++)
        {
            if( stateSpaceQuantization.theta_quantized[i] == 1){
                verify_theta_quantized = true;
            }
        }

        for (int i=0;i < stateSpaceQuantization.theta_dot_quantized.length;i++)
        {
            if( stateSpaceQuantization.theta_dot_quantized[i] == 1){
                verify_theta_dot_quantized = true;
            }
        }

        for (int i=0;i < stateSpaceQuantization.x_quantized.length;i++)
        {
            if( stateSpaceQuantization.x_quantized[i] == 1){
                verify_x_quantized = true;
            }
        }

        for (int i=0;i < stateSpaceQuantization.x_dot_quantized.length;i++)
        {
            if( stateSpaceQuantization.x_dot_quantized[i] == 1){
                verify_x_dot_quantized = true;
            }
        }
        return verify_theta_quantized && verify_theta_dot_quantized && verify_x_quantized && verify_x_dot_quantized;
    }

    @Override
    public String toString() {
        return "theta_quantized = " + Arrays.toString(theta_quantized) + ", theta_dot_quantized = " + Arrays.toString(theta_dot_quantized) + ", x_quantized = " + Arrays.toString(x_quantized) + ", x_dot_quantized = " + Arrays.toString(x_dot_quantized);
    }
}
