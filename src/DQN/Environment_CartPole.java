package src.DQN;

public class Environment_CartPole {
    public double x_current;
    public double u_current;
    public double L = DQN.Length_Of_Stick;
    public double M = DQN.Mass_Of_Cart;
    public double m = DQN.Mass_Of_Stick;
    public double g = DQN.g;
    public double theta;
    public double t = 0;
    public double dt = 0.02;
    public double omega = 0;

    public class State {
        public double x;
        public double x_dot;
        public double theta;
        public double theta_dot;

        public double time;
        public double theta_degree;
        public double reward;

        public State(double x, double x_dot, double theta, double theta_dot, double time) {
            this.x = x;
            this.x_dot = x_dot;
            this.theta = theta;
            this.theta_dot = theta_dot;
            this.time = time;
            theta_degree = (this.theta * (180 / Math.PI));
            StateSpaceQuantization stateSpaceQuantization =  StateSpaceQuantization.getBox1(this);
            if(!StateSpaceQuantization.verify(stateSpaceQuantization)) reward = -1;
            else reward = 0;
        }

        @Override
        public String toString() {
            return "x = " + x + ", x_dot = " + x_dot + ", theta_dot = " + theta_dot + ", theta = " + theta+ ", reward = " + reward + ", time = " + time + ", Theta degree = " + theta_degree;
        }
    }

    public Environment_CartPole(double O_Initial){
        this.theta = O_Initial;
    }

    public State getNewStateAndReward(double force){
        //System.out.println("force " + force);
        //double angular_acc = (Math.sin(theta) * (g/L)) - (acceleration * (Math.cos(theta)/L)) ;

        // Open AI cartpole
/*        double temp  = (
                force + (0.5 * 0.1) * Math.pow(omega,2) * Math.sin(theta)
        ) / (M+m);
        double angular_acc  = (g * Math.sin(theta) - Math.cos(theta) * temp) / (
                0.5 * (4.0 / 3.0 - m *  Math.pow(Math.cos(theta),2) / (M+m))
        );
        double cart_acceleration = temp - (0.5 * 0.1) * angular_acc * Math.cos(theta) / (M+m);*/

        //double angular_acc = ( ( (M+m)*g*Math.sin(theta)) - ( (Math.cos(theta) * ( force + (m * L * Math.pow(omega,2)*Math.sin(theta)) ) ) ) )/( ( (M+m) * L) - ( m * L * Math.pow(Math.cos(theta),2)));
        //double cart_acceleration = (((m * angular_acc) - (g*Math.sin(theta)))/(Math.cos(theta)));

        double angular_acc = ( ( (M+m)*g*Math.sin(theta)) - ( (Math.cos(theta) * ( force + (m * L * Math.pow(omega,2)*Math.sin(theta)) ) ) ) )/( ((4/3.0) * (M+m) * L) - ( m * L * Math.pow(Math.cos(theta),2)));
        double cart_acceleration = ( force + (m*L* (  (Math.pow(omega,2)*Math.sin(theta) ) - (angular_acc * Math.cos(theta))) ))/(M+m);

        u_current = u_current + cart_acceleration * dt;
        x_current = x_current + u_current *dt;
        omega = omega + angular_acc * dt;
        theta = theta + omega * dt;
        t= t + dt;

        return new State(x_current, u_current, theta, omega, t);
    }

    public State getState(){
        return new State(x_current, u_current, theta, omega, t);
    }
}
